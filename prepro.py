from dataclasses import dataclass
from typing import List, Optional, Union
import json
import dataclasses

@dataclass
class InputExample:

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class NERInputExample:

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[List[str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"