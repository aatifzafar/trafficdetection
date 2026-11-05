import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.database import Base, engine, SessionLocal
from backend.db_models.models import Violation, CameraSource

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "demo_mode" in data


def test_get_cameras():
    response = client.get("/api/cameras")
    assert response.status_code == 200
    cameras = response.json()
    assert isinstance(cameras, list)
    assert len(cameras) > 0


def test_create_and_query_camera():
    payload = {
        "name": "Test Test Junction Camera",
        "location": "Jayanagar 4th Block, Bengaluru",
        "source_type": "VIDEO_FILE",
        "source_url": "data/videos/test.mp4",
        "latitude": 12.9299,
        "longitude": 77.5824,
        "direction": "Eastbound",
    }
    response = client.post("/api/cameras", json=payload)
    assert response.status_code == 201
    created = response.json()
    assert created["name"] == payload["name"]
    cam_id = created["id"]

    # Start camera
    res_start = client.post(f"/api/cameras/{cam_id}/start")
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "ONLINE"

    # remove the test camera again, so test runs do not fill the real camera list
    db = SessionLocal()
    try:
        db.query(CameraSource).filter(CameraSource.id == cam_id).delete()
        db.commit()
    finally:
        db.close()


def test_analytics_summary():
    response = client.get("/api/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_violations" in data
    assert "active_cameras" in data


def test_database_violation_insertion():
    import uuid
    db = SessionLocal()
    unique_event = f"EVT-TEST-{uuid.uuid4().hex[:8]}"
    unique_plate = f"KA{uuid.uuid4().hex[:2].upper()}TEST"
    try:
        test_viol = Violation(
            event_id=unique_event,
            violation_type="NO_HELMET",
            track_id=101,
            plate_number=unique_plate,
            confidence=0.92,
            location="Test Junction, Bengaluru",
            status="CONFIRMED",
        )
        db.add(test_viol)
        db.commit()

        # Query via API
        response = client.get(f"/api/violations?plate_number={unique_plate}")
        assert response.status_code == 200
        viols = response.json()
        assert len(viols) >= 1
        assert viols[0]["event_id"] == unique_event
        viol_id = viols[0]["id"]

        # Test challan generation
        res_ch = client.get(f"/api/violations/{viol_id}/challan")
        assert res_ch.status_code == 200
        challan_data = res_ch.json()
        assert "PROTO-" in challan_data["challan_number"]
        assert challan_data["fine_amount_inr"] == 1000
        assert "Section 194D" in challan_data["mv_act_section"]

        # Test challan issuance
        res_issue = client.post(f"/api/violations/{viol_id}/issue-challan")
        assert res_issue.status_code == 200
        assert res_issue.json()["challan"]["violation_id"] == viol_id

        # Verify status updated
        db.expire_all()
        v_check = db.query(Violation).filter(Violation.id == viol_id).first()
        assert v_check.status == "APPROVED"
    finally:
        db.close()



def test_review_workflow_requires_reason_and_records_reviewer():
    import uuid
    db = SessionLocal()
    try:
        v = Violation(event_id=f"EVT-REVIEW-{uuid.uuid4().hex[:8]}", violation_type="TRIPLE_RIDING",
                      confidence=0.8, location="Review Junction")
        db.add(v)
        db.commit()
        assert v.status == "PENDING_REVIEW"
        vid = v.id

        r = client.post(f"/api/violations/{vid}/review", json={"action": "reject", "reviewer": "Insp. Test"})
        assert r.status_code == 400  # reason is mandatory

        r = client.post(f"/api/violations/{vid}/review",
                        json={"action": "reject", "reviewer": "Insp. Test", "reason": "Not a violation"})
        assert r.status_code == 200
        assert r.json()["status"] == "REJECTED"
        assert r.json()["reviewed_by"] == "Insp. Test"
    finally:
        db.close()
