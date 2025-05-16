# audit_app/models_ui.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

# Attempt to import from AssessmentSystem.model. If running as a script directly, this might fail,
# so for standalone execution, you might need to adjust paths or ensure AssessmentSystem is in PYTHONPATH.
try:
    from AssessmentSystem.model import Judgement, EvidenceSearchParams, Conclusion as CoreConclusion, EvidenceSearchResult as CoreEvidenceSearchResult
except ImportError:
    # Fallback for cases where the module might not be found (e.g. direct script run without proper packaging)
    # This is a simplified version for fallback. In a real packaged app, the import should work.
    print("Warning: Could not import from AssessmentSystem.model. Using fallback basic types.")
    class Judgement(str, Enum):
        COMPLIANT = "符合"
        NON_COMPLIANT = "不符合"
        NOT_APPLICABLE = "不涉及"
        NOT_PROCESSED = "未处理"
        ERROR = "error"

    class EvidenceSearchParams(BaseModel):
        query_text: str
        filter_tags: List[str] = []
        terminated: bool = False

    class CoreConclusion(BaseModel):
        judgement: Judgement = Judgement.NOT_PROCESSED
        comment: str = ""
        evidence: List[Dict[str, Any]] = [] # Simplified for fallback

    class CoreEvidenceSearchResult(BaseModel):
        source: str = "unknown"
        content: str = ""
        tags: Optional[List[str]] = None


class ItemAuditStatus(str, Enum):
    """Tracks the auditor's progress for each assessment item."""
    NOT_REVIEWED = "未审核"
    SKIPPED = "已跳过"
    REVIEWED = "已审核"

class UIEvidence(BaseModel):
    """
    Represents an evidence item in the UI.
    It can be an evidence found from a search, or an evidence referenced in an assessment item.
    """
    evidence_id: str = Field(description="Unique identifier for the evidence. Typically derived from 'source' or a dedicated 'id' field from the backend.")
    content: str = Field(description="Full content of the evidence.")
    short_content: str = Field(description="A short preview of the content for display in cards, e.g., first 200 chars.")

    # Optional fields from requirements, might not be in all data sources
    title: Optional[str] = None
    evidence_type: Optional[str] = None # E.g., "log", "screenshot", "document"
    timestamp_str: Optional[str] = None # Evidence timestamp as a string

    # State for referenced evidences
    is_active_for_conclusion: bool = Field(True, description="If referenced, is it 'active' (supporting conclusion) or 'suppressed' (user marked as not relevant, greyed out).")

    # For search results, associated tags might be provided
    search_tags: Optional[List[str]] = Field(default_factory=list, description="Tags associated with this evidence from search results, if any.")

    class Config:
        frozen = False # Allow fields to be mutable for UI state changes like is_active_for_conclusion

    @classmethod
    def from_core_search_result(cls, core_result: CoreEvidenceSearchResult) -> 'UIEvidence':
        """
        Creates a UIEvidence instance from a CoreEvidenceSearchResult object.
        The 'source' field of CoreEvidenceSearchResult is assumed to be the evidence_id.
        """
        content = core_result.content
        # Assuming core_result.source is the unique ID. If NaviSearchClient's _get_source_from_record
        # logic (which checks 'id' then 'source') is used to populate CoreEvidenceSearchResult.source, this is fine.
        evidence_id = str(core_result.source)

        return cls(
            evidence_id=evidence_id,
            content=content,
            short_content=content[:200] + "..." if len(content) > 200 else content,
            search_tags=getattr(core_result, 'tags', []) # Get tags if available
        )

    @classmethod
    def from_report_evidence_dict(cls, evidence_dict: Dict[str, Any]) -> 'UIEvidence':
        """
        Creates a UIEvidence instance from an evidence dictionary as found in
        the 'assessment_report_example.json' under 'conclusion.evidence'.
        The 'source' field in this dict is treated as the evidence_id.
        """
        evidence_id = str(evidence_dict.get("source", f"unknown_id_{hash(evidence_dict.get('content', ''))}"))
        content = evidence_dict.get("content", "")
        return cls(
            evidence_id=evidence_id,
            content=content,
            short_content=content[:200] + "..." if len(content) > 200 else content,
            is_active_for_conclusion=True # Referenced evidences from report start as active
        )

class UIAssessmentItem(BaseModel):
    """Represents a single assessment item in the UI, including its audit state."""
    spec_id: str
    spec_content: str

    # Data loaded from the original assessment report
    initial_search_params: Optional[EvidenceSearchParams] = None
    initial_conclusion: Optional[CoreConclusion] = None # Original judgement, comment, and evidence list (as dicts)
    ai_assessment_status: Optional[str] = Field(None, description="Original AI assessment status, e.g., 'success', 'error'.")
    ai_error_message: Optional[str] = Field(None, description="Original AI error message, if any.")

    # Current audit state being manipulated by the user
    current_judgement: Judgement = Judgement.NOT_PROCESSED
    current_comment: str = ""
    # List of UIEvidence objects that are currently referenced for this assessment item.
    # Their 'is_active_for_conclusion' status determines if they are active or suppressed.
    referenced_evidences: List[UIEvidence] = Field(default_factory=list)

    # State for LLM suggestions comparison
    llm_suggested_judgement: Optional[Judgement] = None
    llm_suggested_comment: Optional[str] = None
    # If LLM suggests specific evidences to keep, their IDs can be stored here.
    # For now, the logic is: LLM comments on currently *active* evidences.
    # If suggestion accepted, *suppressed* evidences are removed.

    audit_status: ItemAuditStatus = ItemAuditStatus.NOT_REVIEWED

    # Internal: Store original evidence dicts from the report for reset or reference
    raw_initial_evidences: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)

    class Config:
        frozen = False # Allow fields to be mutable

class UIAssessmentReport(BaseModel):
    """Top-level Pydantic model representing the entire assessment report for the UI."""
    report_metadata: Dict[str, Any] = Field(default_factory=dict, description="Stores original top-level keys from the report, e.g., 'statics'.")
    assessment_items: List[UIAssessmentItem] = Field(default_factory=list)

    # UI and session state
    current_item_index: int = 0
    original_file_path: Optional[str] = None
    reviewed_file_path: Optional[str] = None # Target path for saving the reviewed report

    # Display statistics
    stats_judgement: Dict[str, int] = Field(default_factory=dict) # Keyed by Judgement.value
    stats_audit_status: Dict[str, int] = Field(default_factory=dict) # Keyed by ItemAuditStatus.value

    class Config:
        frozen = False # Allow fields to be mutable

    def update_stats(self):
        """Recalculates statistics based on the current state of assessment items."""
        self.stats_judgement = {j.value: 0 for j in Judgement}
        self.stats_audit_status = {s.value: 0 for s in ItemAuditStatus}

        for item in self.assessment_items:
            self.stats_judgement[item.current_judgement.value] = self.stats_judgement.get(item.current_judgement.value, 0) + 1
            self.stats_audit_status[item.audit_status.value] = self.stats_audit_status.get(item.audit_status.value, 0) + 1
        return self # Return self for chaining or direct use in Gradio State

    def get_current_assessment_item(self) -> Optional[UIAssessmentItem]:
        """Returns the current assessment item based on current_item_index."""
        if 0 <= self.current_item_index < len(self.assessment_items):
            return self.assessment_items[self.current_item_index]
        return None

    def set_current_assessment_item_audit_data(self, judgement: Judgement, comment: str, referenced_evidences: List[UIEvidence]):
        """Updates the current assessment item's audit data."""
        item = self.get_current_assessment_item()
        if item:
            item.current_judgement = judgement
            item.current_comment = comment
            item.referenced_evidences = referenced_evidences # Assumes this list has correct is_active_for_conclusion states
            # Mark as reviewed if a judgement is made (unless it's still 'NOT_PROCESSED')
            if item.audit_status == ItemAuditStatus.NOT_REVIEWED and judgement != Judgement.NOT_PROCESSED:
                 item.audit_status = ItemAuditStatus.REVIEWED
            self.update_stats()

