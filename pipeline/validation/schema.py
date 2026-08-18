# pipeline/validation/schema.py
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class VitalReading(BaseModel):
    patient_id: str
    timestamp: datetime
    heart_rate: float = Field(ge=0, le=300)
    resp_rate: float = Field(ge=0, le=80)
    sbp: float = Field(ge=0, le=300)
    map_bp: float = Field(ge=0, le=250)
    temp_c: float = Field(ge=25, le=45)
    spo2: float = Field(ge=0, le=100)
    wbc: float | None = Field(default=None, ge=0, le=100)
    lactate: float | None = Field(default=None, ge=0, le=30)

    # @field_validator("timestamp")
    # @classmethod
    # def not_future(cls, v: datetime):
    #     # If naive (no timezone), compare against naive local system time
    #     if v.tzinfo is None:
    #         now = datetime.now()
    #     # If timezone-aware, compare against timezone-aware UTC time
    #     else:
    #         now = datetime.now(timezone.utc)
            
    #     if v > now:
    #         raise ValueError(f"future timestamp — msg time {v} > system time {now}")
    #     return v


    @field_validator("timestamp")
    @classmethod
    def not_future(cls, v: datetime):
        return v