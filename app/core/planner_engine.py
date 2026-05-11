from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple, Optional
import math

from app.models.project_member import ProjectMember
from app.models.task import Task
from app.core.summarizer import generate_sprint_summary

# ------------------------------------------------------
# CONFIG CONSTANTS (tune these weights for behaviour)
# ------------------------------------------------------
RELIABILITY_THRESHOLD = 0.50       # Members below this are flagged in risk analysis
SAFETY_BUFFER_PERCENT = 0.10       # 10% safety buffer for capacity
DEADLINE_URGENCY_DAYS = 5          # Tasks due in N days or less are 'Deadline-Critical'
OVERLOAD_PENALTY_THRESHOLD = 0.95  # Capacity utilization > 95% triggers overload risk
TASK_DEFAULT_EFFORT_HOURS = 8.0    # Default effort for a task if not defined

# Fairness weights (sum not required; used proportionally)
W_RELIABILITY = 0.35
W_VELOCITY = 0.35
W_OVERLOAD = 0.15
W_AVAILABILITY = 0.15

# Priority scoring weights
W_URGENCY = 0.35
W_PRIORITY = 0.30
W_BUSINESS_VALUE = 0.15
W_COMPLEXITY = 0.10
W_DEPENDENCY_DEPTH = 0.05
W_DEADLINE_PRESSURE = 0.05

# Fairness slack hours (allow small deviation)
FAIRNESS_SLACK_HOURS = 2.0

# Burndown smoothing (simple)
BURNDOWN_DAYS_SMOOTH = 3

# ------------------------------------------------------
# Helper Functions
# ------------------------------------------------------
def _get_corrected_effort(task: Task) -> float:
    """Returns task effort, defaulting to 8.0h if input is 0 or missing."""
    return task.estimatedHours if getattr(task, "estimatedHours", None) is not None and task.estimatedHours > 0.0 else TASK_DEFAULT_EFFORT_HOURS

def _get_task_id(task: Task) -> str:
    """Safely gets the task ID."""
    return task.taskId

def _calculate_working_days(start: date, end: date, unavailable_dates: List[date]) -> int:
    """
    Calculate the number of working days between two dates, 
    excluding weekends and specific unavailable dates.
    """
    current_date = start
    working_days = 0
    # Add one day to end date to make it inclusive
    while current_date <= end:
        # Check if it's a weekday (Monday=0 to Friday=4)
        if current_date.weekday() < 5:
            # Check if current date is explicitly unavailable
            if current_date not in unavailable_dates:
                 working_days += 1
        current_date += timedelta(days=1)
    return working_days

def _days_until(date_obj: Optional[date], from_date: date) -> Optional[int]:
    if not date_obj:
        return None
    return (date_obj - from_date).days

class SprintPlanner:
    
    def __init__(self, project_id: str, members: List[ProjectMember], tasks: List[Task], sprint_config: Dict[str, Any], max_tasks_per_member: Optional[int] = None):
        self.project_id = project_id
        
        # Members MUST be mapped by ProjectMemberId (the ID received in the JSON body)
        self.members = {m.projectMemberId: m for m in members}
        self.tasks = {t.taskId: t for t in tasks}
        
        print(f"🟦 [PLANNER INIT] Project={project_id} | Members={len(self.members)} | Tasks={len(self.tasks)}")

        # Sprint Configuration
        self.sprint_config = sprint_config
        self.sprint_length_days = sprint_config.get('sprintLengthDays', 14)
        self.work_hours_per_day = sprint_config.get('workHoursPerDay', 8)
        self.max_tasks_per_member = max_tasks_per_member
        
        # State Trackers
        self.total_team_capacity: float = 0.0
        self.member_capacities: Dict[str, float] = {} # Total effective capacity
        self.member_current_load: Dict[str, float] = {mid: 0.0 for mid in self.members} # Assigned load
        self.member_task_count: Dict[str, int] = {mid: 0 for mid in self.members}
        
        # Fairness results
        self.member_fairness_score: Dict[str, float] = {}
        self.member_normalized_fair_share: Dict[str, float] = {} # fraction
        self.member_fair_share_hours: Dict[str, float] = {}

        self.selected_tasks: List[Dict[str, Any]] = []
        self.deferred_tasks: List[Dict[str, Any]] = []
        self.risk_analysis: Dict[str, Any] = {}
        self.recommendations: List[str] = []
        
        self.sprint_start_date = date.today()
        self.sprint_end_date = self.sprint_start_date + timedelta(days=self.sprint_length_days)

    
    # --- 1. Sprint Capacity Calculation ---
    def _calculate_sprint_capacity(self):
        self.total_team_capacity = 0.0
        
        for member_id, member in self.members.items():
            
            # Convert unavailable dates to date objects if they are strings
            member_unavailable_dates = [
                d if isinstance(d, date) else datetime.strptime(str(d).split('T')[0], "%Y-%m-%d").date() 
                for d in getattr(member, "unavailableDates", []) or []
            ]
            
            # Base working hours in the sprint (excluding weekends and explicit unavailability)
            working_days = _calculate_working_days(
                self.sprint_start_date, self.sprint_end_date, member_unavailable_dates
            )
            base_sprint_hours = working_days * self.work_hours_per_day
            
            # Apply all factors from the JSON body
            # NOTE: mapping names to your ProjectMember model attributes
            availability_factor = getattr(member, "availabilityFactor", 1.0)
            reliability_score = getattr(member, "reliabilityScore", 1.0)
            skill_eff = getattr(member, "skillEfficiencyMultiplier", 1.0)
            overload_risk = getattr(member, "overloadRiskScore", 0.0)
            
            effective_hours = (
                base_sprint_hours
                * availability_factor 
                * reliability_score
                * skill_eff
            )
            
            # Adjust for overloadRiskScore (e.g., reduce capacity if risk is already high)
            effective_hours = effective_hours * (1.0 - overload_risk)
            
            # Apply Safety Buffer (10%)
            member_capacity = effective_hours * (1.0 - SAFETY_BUFFER_PERCENT)
            
            member.sprintCapacityHours = max(0.0, member_capacity)
            self.member_capacities[member_id] = member.sprintCapacityHours
            self.total_team_capacity += member_capacity
            
            # For the 1-task-per-member rule, we set effective_max_tasks = 1, unless maxTasksPerMember > 1
            member.effective_max_tasks = self.max_tasks_per_member if self.max_tasks_per_member is not None and self.max_tasks_per_member > 0 else 1
            
            print(f"CapCalc: {member.name} ({member_id}): Base={base_sprint_hours}h. Effective={member_capacity:.1f}h. MaxTasks={member.effective_max_tasks}")

    # --- Fairness calculations (new) ---
    def _compute_member_fairness(self):
        """Compute a fairness score per member and normalize into fair share hours."""
        # Prepare statistics
        velocities = [getattr(m, "velocity", 0.0) for m in self.members.values()]
        avg_velocity = sum(velocities) / len(velocities) if velocities else 1.0
        raw_scores = {}
        for mid, m in self.members.items():
            rel = getattr(m, "reliabilityScore", 1.0)
            vel = getattr(m, "velocity", avg_velocity)
            overload = getattr(m, "overloadRiskScore", 0.0)
            avail = getattr(m, "availabilityFactor", 1.0)

            # Compose raw fairness score (higher = better)
            # velocity normalized by average_velocity provides relative throughput
            vel_component = (vel / avg_velocity) if avg_velocity > 0 else 1.0
            raw = (
                (rel * W_RELIABILITY) +
                (vel_component * W_VELOCITY) +
                ((1.0 - overload) * W_OVERLOAD) +
                (avail * W_AVAILABILITY)
            )
            raw_scores[mid] = max(0.0, raw)

        total_raw = sum(raw_scores.values()) or 1.0
        # Normalize into fractions and compute fair share hours
        for mid, raw in raw_scores.items():
            frac = raw / total_raw
            fair_hours = frac * self.total_team_capacity
            self.member_fairness_score[mid] = round(raw, 4)
            self.member_normalized_fair_share[mid] = round(frac, 4)
            self.member_fair_share_hours[mid] = round(fair_hours, 2)

        print("Fairness: ", self.member_fairness_score, self.member_fair_share_hours)

    # --- Utility: compute dependency depth (simple DFS) ---
    def _dependency_depth(self, task: Task, visited=None) -> int:
        if visited is None:
            visited = set()
        depth = 0
        for dep in getattr(task, "dependencies", []) or []:
            if dep in visited:
                continue
            visited.add(dep)
            dep_task = self.tasks.get(dep)
            if dep_task:
                depth = max(depth, 1 + self._dependency_depth(dep_task, visited))
            else:
                depth = max(depth, 1)
        return depth

    # --- 2. Task Eligibility Filtering ---
    def _filter_tasks(self) -> List[Task]:
        """Filters tasks for the sprint based on status, dependencies, and deadlines."""
        completed_task_ids = {t.taskId for t in self.tasks.values() if getattr(t, "status", "") in ["Done", "Completed"]}
        
        eligible_tasks = []
        
        for task in self.tasks.values():
            
            # R1: Not done yet
            if getattr(task, "status", "") not in ["Backlog", "Open"]:
                continue
            
            # R2: Not blocked by dependencies (we will keep blocked tasks as deferred with richer reason)
            blocked_deps = []
            for dep_id in getattr(task, "dependencies", []) or []:
                if dep_id not in completed_task_ids:
                    blocked_deps.append(dep_id)
            
            if blocked_deps:
                self.deferred_tasks.append({
                    "taskId": task.taskId, 
                    "reason": (
                        f"Blocked by dependency(ies): {', '.join(blocked_deps)}. "
                        f"These blocking tasks must be completed before this task can be scheduled. "
                        f"Dependency count: {len(blocked_deps)}."
                    )
                })
                continue
            
            # R3: Assigned member check (Only defer if assigned to an UNKNOWN member)
            assignee = getattr(task, "assignedTo", None)
            # The task.assignedTo may be nested object (older schema) — check for id
            if isinstance(assignee, dict):
                assignee_id = assignee.get("_id") or assignee.get("id")
            else:
                assignee_id = assignee

            task._assignee_id_resolved = assignee_id  # attach for later use

            if assignee_id and assignee_id not in self.members:
                self.deferred_tasks.append({
                    "taskId": task.taskId, 
                    "reason": (
                        f"Assigned to unknown member ID: {assignee_id}. "
                        f"Please ensure the assigned member exists in project members or remove assignment."
                    )
                })
                continue
                
            # R4: Deadline check — tag eligibility reason
            task.eligibility_reason = ""
            if getattr(task, "deadline", None):
                time_to_deadline = (task.deadline - self.sprint_start_date).days
                if time_to_deadline <= DEADLINE_URGENCY_DAYS:
                    task.eligibility_reason = "Eligible + Deadline-Critical"
            
            eligible_tasks.append(task)
            
        return eligible_tasks

    # --- Priority scoring with deadline-pressure & dependency depth (new) ---
    def _compute_task_priority_score(self, task: Task) -> float:
        """
        Combine urgency, static priority, business value, complexity, dependency depth, and deadline pressure.
        Returns a numeric score (higher = more important).
        """
        # Priority (High/Med/Low)
        priority_map = {"High": 3, "Medium": 2, "Low": 1}
        pval = priority_map.get(getattr(task, "priority", None), 1)

        # Business value (if present)
        bv = float(getattr(task, "businessValue", 0) or 0)

        # Complexity mapping (higher complexity reduces score slightly because high complexity is harder)
        complexity_map = {"Low": 1, "Medium": 2, "High": 3}
        cval = complexity_map.get(getattr(task, "complexity", None), 2)

        # Deadline pressure: inverse days until deadline (more pressure if fewer days)
        days_until_deadline = _days_until(getattr(task, "deadline", None), self.sprint_start_date)
        deadline_pressure = 0.0
        if days_until_deadline is not None:
            # If negative (already past), treat as maximum pressure
            days = max(0, days_until_deadline)
            deadline_pressure = 1.0 / (days + 1)  # ranges (1 for due today) -> small for distant

        # Dependency depth
        dep_depth = self._dependency_depth(task)

        # Compose weighted score (scale terms appropriately)
        score = (
            (W_PRIORITY * (pval / 3.0)) +
            (W_BUSINESS_VALUE * (bv / (bv + 1) if bv >= 0 else 0)) +
            (W_URGENCY * (1.0 if "Deadline-Critical" in getattr(task, "eligibility_reason", "") else 0.0)) +
            (W_COMPLEXITY * (1.0 - (cval / 3.0))) +  # prefer smaller complexity slightly
            (W_DEPENDENCY_DEPTH * min(1.0, dep_depth / 5.0)) +
            (W_DEADLINE_PRESSURE * deadline_pressure)
        )
        # Multiply by a simple scale to get a useful numeric ranking
        return round(score * 100.0, 4)

    # --- 3. Intelligent Sprint Selection (One-Task-Per-Member Strategy) with Fairness ---
    def _select_tasks(self, eligible_tasks: List[Task]):
        # Precompute fairness (requires capacities)
        self._compute_member_fairness()

        # 1. Group tasks by their current assignment state
        tasks_by_assignee = {}
        unassigned_tasks = []
        for task in eligible_tasks:
            assignee_id = getattr(task, "_assignee_id_resolved", None)
            if not assignee_id:
                unassigned_tasks.append(task)
            else:
                tasks_by_assignee.setdefault(assignee_id, []).append(task)
        
        # 2. Compute priority scores for all tasks and sort
        for pool in (unassigned_tasks,):
            for t in pool:
                t._priority_score = self._compute_task_priority_score(t)
        for mid, pool in tasks_by_assignee.items():
            for t in pool:
                t._priority_score = self._compute_task_priority_score(t)
        
        unassigned_tasks.sort(key=lambda t: getattr(t, "_priority_score", 0.0), reverse=True)
        for mid in tasks_by_assignee:
             tasks_by_assignee[mid].sort(key=lambda t: getattr(t, "_priority_score", 0.0), reverse=True)

        selected_task_ids = set()

        # 3. Iterate through members ordered by reliability & availability (same as before)
        sorted_members = sorted(
            self.members.values(),
            key=lambda m: (getattr(m, "reliabilityScore", 0.0), getattr(m, "availabilityFactor", 0.0)),
            reverse=True
        )
        
        for member in sorted_members:
            member_id = member.projectMemberId
            
            # skip if no capacity at all
            if self.member_capacities.get(member_id, 0.0) <= 0:
                self.recommendations.append(f"{member.name} has zero effective capacity this sprint (unavailable or zero factors).")
                continue

            if self.member_task_count[member_id] >= member.effective_max_tasks:
                # Already reached task count
                continue

            best_task = None

            # 3A. Search 1: Highest-priority task pre-assigned to this member
            for task in tasks_by_assignee.get(member_id, []):
                if task.taskId not in selected_task_ids:
                    best_task = task
                    break

            # 3B. (unchanged) fallback: unassigned pool (we won't reassign pre-assigned tasks from others)
            if not best_task:
                 for task in unassigned_tasks:
                     if task.taskId not in selected_task_ids:
                         best_task = task
                         break

            if not best_task:
                self.recommendations.append(f"{member.name} capacity is unused this sprint (no eligible, unselected task found).")
                continue

            task_effort = _get_corrected_effort(best_task)
            remaining_capacity = self.member_capacities.get(member_id, 0.0) - self.member_current_load.get(member_id, 0.0)

            # FAIRNESS DECISION: compare planned hours if this task included to fair share hours
            planned_after = self.member_current_load.get(member_id, 0.0) + task_effort
            fair_share = self.member_fair_share_hours.get(member_id, 0.0)

            # Allow slack
            allowed_threshold = fair_share + FAIRNESS_SLACK_HOURS

            # Decision logic:
            # - enforce capacity first (can't exceed remaining capacity)
            # - then check if planned_after <= allowed_threshold. If it exceeds, defer with fairness reason.
            if remaining_capacity < task_effort:
                # Not enough capacity -> defer with explicit capacity reason
                reason_detail = (
                    f"CAPACITY VIOLATION: Member {member.name} remaining capacity {remaining_capacity:.1f}h is less than task estimated {task_effort:.1f}h. "
                    f"Task skipped for this sprint. Consider reassigning, splitting task, or increasing sprint capacity."
                )
                self.recommendations.append(reason_detail)
                if getattr(best_task, "assignedTo", None):
                    self.deferred_tasks.append({
                        "taskId": best_task.taskId,
                        "reason": reason_detail
                    })
                continue

            # Fairness check: if pre-assigned and would exceed fair share by a lot, then defer
            pre_assigned_to_member = getattr(best_task, "_assignee_id_resolved", None) == member_id and getattr(best_task, "assignedTo", None) is not None

            if pre_assigned_to_member and planned_after > allowed_threshold:
                # Defer with a detailed fairness reason
                reason_detail = (
                    f"FAIRNESS LIMIT: Task is pre-assigned to {member.name} but including it would push their planned hours to {planned_after:.1f}h, "
                    f"which exceeds their fair share ({fair_share:.1f}h) + slack ({FAIRNESS_SLACK_HOURS:.1f}h). "
                    f"Fairness score: {self.member_fairness_score.get(member_id, 0.0)}. "
                    f"This prevents overloading the same member repeatedly across sprints. Consider reassigning or splitting the task."
                )
                self.deferred_tasks.append({
                    "taskId": best_task.taskId,
                    "reason": reason_detail
                })
                # do not assign
                continue

            # If unassigned task (fallback) we still enforce fair share but be more permissive:
            if not pre_assigned_to_member and planned_after > (fair_share + FAIRNESS_SLACK_HOURS * 2):
                reason_detail = (
                    f"FAIRNESS LIMIT (UNASSIGNED): Including this unassigned task would push {member.name}'s planned hours to {planned_after:.1f}h, "
                    f"which is above their fair share ({fair_share:.1f}h). Skipping this unassigned task to maintain fairness."
                )
                self.deferred_tasks.append({
                    "taskId": best_task.taskId,
                    "reason": reason_detail
                })
                continue

            # --- SUCCESSFUL ASSIGNMENT ---
            self.member_current_load[member_id] += task_effort
            self.member_task_count[member_id] += 1
            selected_task_ids.add(best_task.taskId)
            
            # Build a descriptive reason string
            reason_parts = []
            if pre_assigned_to_member:
                reason_parts.append("PRE-ASSIGNED to this member.")
            else:
                reason_parts.append("TEMPORARY assignment from unassigned pool for planning.")
            
            reason_parts.append(f"PriorityScore={getattr(best_task, '_priority_score', 0.0):.2f}.")
            if "Deadline-Critical" in getattr(best_task, "eligibility_reason", ""):
                reason_parts.append("Deadline-critical: requires immediate attention within sprint.")
            reason_parts.append(f"Estimated effort: {task_effort:.1f}h. Member remaining capacity before assignment: {remaining_capacity:.1f}h.")
            reason_parts.append(f"Member fair share hours: {fair_share:.1f}h (fairnessScore={self.member_fairness_score.get(member_id, 0.0)}).")
            reason = " ".join(reason_parts)
            
            member_details_out = {
                "projectMemberId": member.projectMemberId,
                "name": member.name,
                "role": member.role,
                "reliabilityScore": getattr(member, "reliabilityScore", None),
                "effectiveCapacity": round(self.member_capacities.get(member_id, 0.0), 1),
                "currentLoad": round(self.member_current_load.get(member_id, 0.0), 1),
                "fairShareHours": round(self.member_fair_share_hours.get(member_id, 0.0), 1),
                "fairnessScore": self.member_fairness_score.get(member_id)
            }
            if hasattr(best_task, 'assigneeDetails') and isinstance(best_task.assigneeDetails, dict):
                member_details_out.update(best_task.assigneeDetails)
            
            self.selected_tasks.append({
                "taskId": best_task.taskId,
                "estimatedHours": task_effort,
                "assignedTo": member_id, 
                "reason": reason,
                "assignedMemberDetails": member_details_out
            })
        
        # 4. Defer all remaining unselected tasks (with improved reason)
        deferred_ids = {d['taskId'] for d in self.deferred_tasks}
        for mid, tasks in tasks_by_assignee.items():
            for task in tasks:
                if task.taskId not in selected_task_ids and task.taskId not in deferred_ids:
                    assignee_name = self.members.get(mid).name if mid in self.members else str(mid)
                    reason = (
                        f"Deferred: Task is assigned to {assignee_name} but was not selected this sprint. "
                        f"Possible reasons: lower priority compared to other selected tasks, fairness constraints, or 1-task-per-member limit reached."
                    )
                    self.deferred_tasks.append({
                        "taskId": task.taskId,
                        "reason": reason
                    })
        for task in unassigned_tasks:
            if task.taskId not in selected_task_ids and task.taskId not in deferred_ids:
                reason = (
                    f"Deferred (Unassigned): No member selected this unassigned task this sprint due to priority/fairness/capacity constraints. "
                    f"Consider assigning it to a member or splitting the task."
                )
                self.deferred_tasks.append({
                    "taskId": task.taskId, 
                    "reason": reason
                })

    # --- 5. Predictive Risk Analysis and KPI computations (extended) ---
    def _analyze_and_balance(self):
        
        overloaded_members = []
        critical_dependencies = []
        deadline_threats = []
        total_planned_effort = sum(t['estimatedHours'] for t in self.selected_tasks)
        # persist total planned effort for external inspection
        self.total_planned_effort = total_planned_effort
        
        # 5.1 Sprint Delay Risk (Based on capacity utilization)
        capacity_utilization = total_planned_effort / self.total_team_capacity if self.total_team_capacity > 0 else 0
        sprint_delay_risk_percent = min(100, int(capacity_utilization * 100 * 1.25)) 
        
        # 5.2 Member Load Balancing and Reliability Risk
        for member_id, member in self.members.items():
            current_load = self.member_current_load.get(member_id, 0.0)
            total_capacity = self.member_capacities.get(member_id, 0.0)
            
            if total_capacity > 0 and (current_load / total_capacity) > OVERLOAD_PENALTY_THRESHOLD:
                overloaded_members.append(member_id)
                self.recommendations.append(f"Member {member.name} is overloaded ({current_load:.1f}/{total_capacity:.1f}h). The assigned task is large relative to capacity.")
            
            if getattr(member, "reliabilityScore", 1.0) < RELIABILITY_THRESHOLD:
                self.recommendations.append(f"Member {member.name} has low reliability score ({member.reliabilityScore:.2f}). Consider pairing or reducing workload.")
        
        # 5.3 Dependency/Deadline Risk
        selected_task_ids = {t['taskId'] for t in self.selected_tasks}
        for t in self.selected_tasks:
            task_obj = self.tasks.get(t['taskId'])
            # Dependency Risk
            for dep_id in getattr(task_obj, "dependencies", []) or []:
                dep_task = self.tasks.get(dep_id)
                if dep_task and dep_id not in selected_task_ids and getattr(dep_task, "status", "") not in ["Done", "Completed"]:
                     critical_dependencies.append(dep_id)
                     self.recommendations.append(f"Task {getattr(task_obj, 'title', 'N/A')} depends on uncompleted task {dep_id}.")
            # Deadline Threat
            if getattr(task_obj, "deadline", None):
                time_to_deadline = (task_obj.deadline - self.sprint_end_date).days
                if time_to_deadline <= 0:
                    deadline_threats.append(t['taskId'])
                    self.recommendations.append(f"Task {getattr(task_obj, 'title', 'N/A')} may miss its deadline (due on or before sprint end).")
        
        # 5.4 Sprint KPIs (predicted velocity + burndown forecast + sprint risk score + fairness report)
        predicted_velocity = self._predict_velocity()
        burndown = self._generate_burndown_forecast(total_planned_effort, predicted_velocity)
        sprint_risk_score = self._compute_sprint_risk_score(len(self.deferred_tasks), len(set(critical_dependencies)), len(overloaded_members), len(deadline_threats))
        
        fairness_report = []
        for mid in self.members:
            fairness_report.append({
                "projectMemberId": mid,
                "fairnessScore": self.member_fairness_score.get(mid),
                "normalizedShare": self.member_normalized_fair_share.get(mid),
                "fairShareHours": self.member_fair_share_hours.get(mid),
                "plannedHours": round(self.member_current_load.get(mid, 0.0), 1),
                "overloadFlag": mid in overloaded_members
            })

        # Build member workload summary: task counts and total story points/estimated hours
        member_workload_summary = []
        # Map selected tasks per member
        workload_map: Dict[str, Dict[str, float]] = {mid: {"taskCount": 0, "totalEstimatedHours": 0.0, "totalStoryPoints": 0.0} for mid in self.members}
        for t in self.selected_tasks:
            mid = t.get('assignedTo')
            if mid not in workload_map:
                # Skip unknown members
                continue
            workload_map[mid]["taskCount"] += 1
            est = float(t.get('estimatedHours', 0.0) or 0.0)
            workload_map[mid]["totalEstimatedHours"] += est
            # If task object has explicit storyPoints or points, try to use it (fallback: est/4)
            task_obj = self.tasks.get(t['taskId'])
            sp = None
            if task_obj:
                sp = getattr(task_obj, 'storyPoints', None) or getattr(task_obj, 'points', None)
            if sp is None:
                # derive from hours (conservative mapping: 4 hours = 1 story point)
                sp = round(est / 4.0, 2)
            workload_map[mid]["totalStoryPoints"] += float(sp)

        for mid, vals in workload_map.items():
            member_workload_summary.append({
                "memberId": mid,
                "taskCount": vals["taskCount"],
                "totalEstimatedHours": round(vals["totalEstimatedHours"], 2),
                "totalStoryPoints": round(vals["totalStoryPoints"], 2)
            })
        # persist for external access
        self.member_workload_summary = member_workload_summary

        self.risk_analysis = {
            "delayRiskPercent": sprint_delay_risk_percent,
            "overloadedMembers": list(set(overloaded_members)),
            "criticalDependencies": list(set(critical_dependencies)),
            "deadlineThreats": list(set(deadline_threats))
        }

        # Attach KPIs & fairness to recommendations/state for later output
        self._kpis = {
            "predictedVelocity": round(predicted_velocity, 2),
            "burndownForecast": burndown,
            "sprintRiskScore": round(sprint_risk_score, 1),
            "fairnessReport": fairness_report
        }

    # --- KPI helpers ---
    def _predict_velocity(self) -> float:
        """
        Predict velocity as a function of member velocities scaled by effective capacity and reliability.
        This is intentionally simple: sum(member.velocity * capacity_ratio * reliability)
        """
        if not self.member_capacities or self.total_team_capacity <= 0:
            return 0.0
        total = 0.0
        for mid, m in self.members.items():
            vel = getattr(m, "velocity", 0.0)
            cap = self.member_capacities.get(mid, 0.0)
            rel = getattr(m, "reliabilityScore", 1.0)
            cap_ratio = cap / self.total_team_capacity if self.total_team_capacity else 0.0
            total += vel * cap_ratio * rel
        return total

    def _generate_burndown_forecast(self, total_planned_effort: float, predicted_velocity: float) -> List[Dict[str, Any]]:
        """
        Very lightweight burndown forecast: distribute planned hours over sprint days
        using predicted_velocity as an indicator of burn rate. Returns list of {date, remainingHours}.
        """
        days = self.sprint_length_days
        if days <= 0:
            return []
        # approximate daily burn: scale predicted_velocity to hours/day (predicted_velocity ~ story points).
        # We'll map predicted_velocity to a hours-per-day burn rate relative to total capacity.
        avg_daily_burn = total_planned_effort / days  # naive baseline
        # Smooth small fluctuations
        forecast = []
        remaining = total_planned_effort
        for i in range(days + 1):
            d = self.sprint_start_date + timedelta(days=i)
            forecast.append({"date": d.isoformat(), "remainingHours": round(max(0.0, remaining), 2)})
            # decrement
            if i < days:
                # Use avg_daily_burn but slightly adjust by predicted_velocity / (sum velocities)
                remaining -= avg_daily_burn
        return forecast

    def _compute_sprint_risk_score(self, deferred_count: int, critical_deps: int, overloaded_count: int, deadline_threats: int) -> float:
        """
        Combined sprint risk: weights on deferred tasks, critical dependencies, overloaded members, deadline threats.
        Returns 0..100
        """
        score = 0.0
        score += min(40, deferred_count * 2.0)  # deferred tasks penalize
        score += min(30, critical_deps * 5.0)
        score += min(20, overloaded_count * 7.0)
        score += min(10, deadline_threats * 5.0)
        return min(100.0, score)

    # --- 6. Final Sprint Plan Output ---
    async def get_final_plan(self):

        self._calculate_sprint_capacity()
        eligible_tasks = self._filter_tasks()
        self._select_tasks(eligible_tasks)
        self._analyze_and_balance()

        # Generate Summary asynchronously based on selected tasks
        try:
            summary_data = await generate_sprint_summary([
                self.tasks[t['taskId']].model_dump()
                for t in self.selected_tasks
            ])
        except Exception as e:
            import traceback
            print("🛑 [SUMMARY ERROR]", e)
            print(traceback.format_exc())
            # fall back to a minimal summary object to avoid breaking the response
            summary_data = {
                "aiSummary": "Sprint planned.",
                "aiConfidence": 0.0,
                "goals": self.sprint_config.get('sprintGoals', []),
                "startDate": None,
                "endDate": None,
                "project": None,
                "velocity": None,
                "riskAnalysis": {}
            }

        # If summarizer didn't supply a project, prefer sprint_config.projectName or project_id
        project_name = summary_data.get('project') or self.sprint_config.get('projectName') or self.project_id

        # Merge AI-generated recommendations (if present) into planner recommendations
        ai_recs = summary_data.get('recommendations') or summary_data.get('recommendations', [])
        if isinstance(ai_recs, list) and ai_recs:
            # Only append items that are not already present
            for r in ai_recs:
                if r not in self.recommendations:
                    self.recommendations.append(r)

        member_capacities_output = [
            {"projectMemberId": mid, "effectiveHours": round(cap, 1)}
            for mid, cap in self.member_capacities.items()
        ]
        
        # Build final output with KPIs and fairness
        output = {
            "success": True,
            "sprintId": f"SPRINT-{datetime.now().strftime('%Y%m%d%H%M')}",
            "summary": summary_data.get("aiSummary", "Optimal sprint scheduled based on capacity and priority."),
            "goals": summary_data.get("goals", self.sprint_config.get("sprintGoals", [])),
            # AI metadata and context fields (populated by summarizer)
            "aiSummary": summary_data.get("aiSummary"),
            "aiConfidence": summary_data.get("aiConfidence"),
            "startDate": summary_data.get("startDate"),
            "endDate": summary_data.get("endDate"),
            "project": project_name,
            "velocity": summary_data.get("velocity") or self._kpis.get("predictedVelocity"),
            "capacity": {
                "totalCapacityHours": round(self.total_team_capacity, 1),
                "memberCapacities": member_capacities_output
            },
            "selectedTasks": self.selected_tasks, 
            "deferredTasks": self.deferred_tasks,
            "riskAnalysis": self.risk_analysis,
            "recommendations": self.recommendations,
            # NEW KPIs & fairness
            "predictedVelocity": self._kpis.get("predictedVelocity"),
            "burndownForecast": self._kpis.get("burndownForecast"),
            "sprintRiskScore": self._kpis.get("sprintRiskScore"),
            "fairnessReport": self._kpis.get("fairnessReport")
        }

        # Add assignment strategy and workload summaries for client consumption
        output["assignmentStrategy"] = "fairness"
        output["memberWorkloadSummary"] = getattr(self, 'member_workload_summary', [])
        output["totalEffort"] = round(getattr(self, 'total_planned_effort', 0.0), 2)

        return output

# Public entry point wraps the class
async def plan_single_sprint(project_id, members, tasks, sprint_config, max_tasks_per_member: Optional[int] = None, debug_mode: bool = False):
    planner = SprintPlanner(project_id, members, tasks, sprint_config, max_tasks_per_member)
    return await planner.get_final_plan()
