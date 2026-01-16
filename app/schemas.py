from pydantic import BaseModel, Field
from typing import Optional, Literal

class StudentInput(BaseModel):
    # Categorical
    school: Literal['GP', 'MS'] = Field(..., description="Student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)")
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
    Medu: int = Field(..., ge=0, le=4, description="Mother's education (0 - none, 1 - primary (4th grade), 2 - 5th to 9th grade, 3 - secondary, 4 - higher)")
    Fedu: int = Field(..., ge=0, le=4, description="Father's education (0 - none, 1 - primary (4th grade), 2 - 5th to 9th grade, 3 - secondary, 4 - higher)")
    traveltime: int = Field(..., ge=1, le=4, description="Home to school travel time (1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time (1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures (numeric: n if 1<=n<3, else 4)")
    freetime: int = Field(..., ge=1, le=5, description="Free time after school (numeric: from 1 - very low to 5 - very high)")
    goout: int = Field(..., ge=1, le=5, description="Going out with friends (numeric: from 1 - very low to 5 - very high)")
    absences: int = Field(..., ge=0, le=93, description="Number of school absences (numeric: 0 to 93)")
    
    # Previous grades (Optional/Required?) - usually known before final prediction
    G1: int = Field(..., ge=0, le=20, description="First period grade")
    G2: int = Field(..., ge=0, le=20, description="Second period grade")

class PredictionOutput(BaseModel):
    prediction_score: float
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
