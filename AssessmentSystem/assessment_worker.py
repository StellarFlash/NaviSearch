# assessment_worker.py
import time
import json
from attr import dataclass
from openai import OpenAI
from concurrent.futures import TimeoutError as FuturesTimeoutError
from utils import get_response
from AssessmentSystem.model import AssessmentSpecItem, AssessmentStatus, Judgement, EvidenceSearchParams, EvidenceSearchResult, Conclusion, AssessmentResult
from AssessmentSystem.llm_client import LLMAssessmentClient
from AssessmentSystem.navi_search_client import NaviSearchClient

class AssessmentWorker:
    def __init__(self, navisearch_client, llm_client, timeout_seconds=500):
        self.navisearch_client = navisearch_client
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds

    def _get_relevant_evidences(self, spec_item: AssessmentSpecItem):
        search_params = self.llm_client.generate_search_params(spec_item)
        evidences_found = self.navisearch_client.search_evidence(search_params)
        # This will call the /search endpoint of VisitorFastAPI
        return evidences_found

    def process_task(self, spec_item) -> dict:
        start_time = time.time()
        try:
            # 1. Search for evidence
            # Using a simple timeout mechanism for the whole process_task
            # More granular timeouts for search and LLM calls might be needed in practice.
            evidences_found = self._get_relevant_evidences(spec_item)
            if time.time() - start_time > self.timeout_seconds:
                raise FuturesTimeoutError("Evidence search and processing timed out")

            # 2. Call LLM for assessment
            llm_response_data = self.llm_client.generate_assessment(spec_item, evidences_found)
            if time.time() - start_time > self.timeout_seconds:
                raise FuturesTimeoutError("LLM assessment timed out")
            # print("LLM response:", llm_response_data)  # Debugging: print the LLM response for debugging purposes.

            return {
                "spec_id": spec_item.id,
                "spec_content": spec_item.content,
                "evidence": llm_response_data.evidence, # List of dicts with 'source', 'content'
                "conclusion": {
                    "judgement": llm_response_data.judgement,
                    "comment": llm_response_data.comment
                },
                "status": "success"
            }

        except FuturesTimeoutError:
            return {
                "spec_id": spec_item.id,
                "spec_content": spec_item.content,
                "evidence": [],
                "conclusion": {"judgement": "Error", "comment": f"Processing timed out after {self.timeout_seconds} seconds."},
                "status": "timeout"
            }
        except Exception as e:
            # Log the full error e
            return {
                "spec_id": spec_item.id,
                "spec_content": spec_item.content,
                "evidence": [],
                "conclusion": {"judgement": "Error", "comment": f"An error occurred: {str(e)}"},
                "status": "error"
            }