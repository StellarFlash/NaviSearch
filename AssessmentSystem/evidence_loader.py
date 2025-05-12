# spec_loader.py
import json
from typing import List
from AssessmentSystem.model import EvidenceMaterial

class EvidenceLoader:
    """ Loads assessment specifications from a JSONL file. """

    def load_evidences(self, file_path: str) -> List[EvidenceMaterial]:
        """
        Reads a JSONL file where each line is a JSON object representing a spec item.

        Args:
            file_path: Path to the JSONL specification file.

        Returns:
            A list of AssessmentSpecItem objects.
        """
        evidences: List[EvidenceMaterial] = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        evidences.append(EvidenceMaterial(**data))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line: {line}. Error: {e}")
                        # Depending on requirements, might raise exception or skip line
                        continue
                    except Exception as e:
                        print(f"Error processing spec data on line: {line}. Error: {e}")
                        continue
        except FileNotFoundError:
            print(f"Error: Specification file not found at {file_path}")
            raise # Re-raise to indicate a critical failure
        except Exception as e:
            print(f"An unexpected error occurred while reading {file_path}: {e}")
            raise
        return evidences

if __name__ == "__main__":
    loader = EvidenceLoader()
    evidences = loader.load_evidences("AssessmentSystem/evidences.jsonl")
    print(f"Loaded {len(evidences)} specifications.")