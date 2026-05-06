"""Tool modules: thin wrappers over external resources / databases."""

from tools.appointments import book_appointment, list_doctors_for_specialty
from tools.ehr_db import (
    add_or_update_patient,
    find_patient_by_name,
    initialize_ehr,
    list_patients,
)
from tools.medical_search import medical_search
from tools.vector_index import build_index, search_index

__all__ = [
    "book_appointment",
    "list_doctors_for_specialty",
    "add_or_update_patient",
    "find_patient_by_name",
    "initialize_ehr",
    "list_patients",
    "medical_search",
    "build_index",
    "search_index",
]
