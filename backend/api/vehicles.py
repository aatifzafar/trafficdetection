from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.database import get_db
from backend.models.models import Vehicle, Violation
from backend.schemas.schemas import VehicleResponse

router = APIRouter(prefix="/api/vehicles", tags=["Vehicles"])


@router.get("/{plate_number}", response_model=VehicleResponse)
def get_vehicle_history(plate_number: str, db: Session = Depends(get_db)):
    """Search vehicle records and violation history by registration plate number."""
    clean_plate = plate_number.strip().upper()
    vehicle = db.query(Vehicle).filter(Vehicle.plate_number == clean_plate).first()

    if not vehicle:
        # Check if violations exist under this plate directly
        viols = db.query(Violation).filter(Violation.plate_number == clean_plate).all()
        if not viols:
            raise HTTPException(status_code=404, detail=f"No vehicle found matching plate '{clean_plate}'")

        # Create ad-hoc vehicle representation
        return VehicleResponse(
            id=0,
            plate_number=clean_plate,
            vehicle_type="Unknown",
            first_seen=viols[-1].timestamp,
            last_seen=viols[0].timestamp,
            total_violations=len(viols),
            violations=viols,
        )

    # Fetch violations associated with vehicle
    viols = db.query(Violation).filter(Violation.plate_number == clean_plate).order_by(desc(Violation.timestamp)).all()
    vehicle.violations = viols
    return vehicle
