from pydantic import BaseModel, Field
from typing import Optional, Literal

class StudentInput(BaseModel):
    # Categorical
    # school removed as per new input list
    source: Literal['mat', 'por'] = Field(..., description="Subject (mat - Mathematics or por - Portuguese)")
    famsize: Literal['LE3', 'GT3'] = Field(..., description="Family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)")
    reason: Literal['home', 'reputation', 'course', 'other'] = Field(..., description="Reason to choose this school")
    guardian: Literal['mother', 'father', 'other'] = Field(..., description="Student's guardian")
    schoolsup: Literal['yes', 'no'] = Field(..., description="Extra educational support")
    famsup: Literal['yes', 'no'] = Field(..., description="Family educational support")
    paid: Literal['yes', 'no'] = Field(..., description="Extra paid classes within the course subject")
    activities: Literal['yes', 'no'] = Field(..., description="Extra-curricular activities")
    nursery: Literal['yes', 'no'] = Field(..., description="Attended nursery school")
    higher: Literal['yes', 'no'] = Field(..., description="Wants to take higher education")
    internet: Literal['yes', 'no'] = Field(..., description="Internet access at home")

    # Numeric
    # Medu, Fedu removed
    traveltime: int = Field(..., ge=1, le=4, description="Home to school travel time (1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time (1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures (numeric: n if 1<=n<3, else 4)")
    freetime: int = Field(..., ge=1, le=5, description="Free time after school (numeric: from 1 - very low to 5 - very high)")
    # goout, absences removed? Wait, absences IS in the list: 'absences'
    # The user list: ['traveltime' 'studytime' 'failures' 'freetime' 'absences' ...]
    absences: int = Field(..., ge=0, le=93, description="Number of school absences (numeric: 0 to 93)")
    
    # G1, G2 removed

class PredictionOutput(BaseModel):
    prediction_score: Optional[float] = None
    prediction_label: str # "Pass" or "Fail" for classification, or just numeric for regression
    probabilities: Optional[dict] = None
    model_type: str

class LogEntry(BaseModel):
    id: int
    timestamp: str
    inputs: dict
    prediction: dict
    
    class Config:
        orm_mode = True
