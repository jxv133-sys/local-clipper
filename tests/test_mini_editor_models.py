"""Tests for mini video editor data models and session management."""

import time
import pytest
from pipeline.models import (
    EditorSession,
    VerticalFormattingJob,
    SessionStore,
    FacecamRegion,
    CanvasLayout,
)


class TestEditorSession:
    """Tests for EditorSession dataclass."""

    @pytest.fixture
    def sample_facecam_region(self) -> FacecamRegion:
        """Create a sample facecam region."""
        return FacecamRegion(
            x=100,
            y=50,
            width=400,
            height=300,
            corner="top-left",
            confidence=0.85,
        )

    @pytest.fixture
    def sample_canvas_layout(self) -> CanvasLayout:
        """Create a sample canvas layout."""
        return CanvasLayout(
            canvas_width=1080,
            canvas_height=1920,
            facecam_x=0,
            facecam_y=0,
            facecam_width=1080,
            facecam_height=672,
            gameplay_x=0,
            gameplay_y=672,
            gameplay_width=1080,
            gameplay_height=1248,
        )

    @pytest.fixture
    def sample_session(
        self, sample_facecam_region, sample_canvas_layout
    ) -> EditorSession:
        """Create a sample editor session."""
        return EditorSession(
            session_id="test-session-123",
            clip_batch_id="batch-001",
            reference_clip_path="/path/to/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )

    def test_session_creation(self, sample_session):
        """Test that a session is created with correct attributes."""
        assert sample_session.session_id == "test-session-123"
        assert sample_session.clip_batch_id == "batch-001"
        assert sample_session.reference_clip_path == "/path/to/clip.mp4"
        assert sample_session.reference_resolution == (1920, 1080)
        assert sample_session.undo_history == []
        assert sample_session.redo_history == []
        assert sample_session.settings == {}

    def test_session_expiry_not_expired(self, sample_session):
        """Test that a newly created session is not expired."""
        assert not sample_session.is_expired()

    def test_session_expiry_expired(self, sample_session):
        """Test that a session with past expiry time is expired."""
        sample_session.expires_at = time.time() - 100  # 100 seconds in the past
        assert sample_session.is_expired()

    def test_session_refresh_expiry(self, sample_session):
        """Test that refresh_expiry extends the expiry time."""
        old_expiry = sample_session.expires_at
        time.sleep(0.1)  # Small delay to ensure time difference
        sample_session.refresh_expiry(1800)
        assert sample_session.expires_at > old_expiry

    def test_session_refresh_expiry_custom_timeout(self, sample_session):
        """Test that refresh_expiry respects custom timeout."""
        before = time.time()
        sample_session.refresh_expiry(3600)  # 1 hour
        after = time.time()
        # Expiry should be approximately 1 hour from now
        # Allow for small timing differences (within 2 seconds)
        expected_expiry = after + 3600
        assert abs(sample_session.expires_at - expected_expiry) < 2

    def test_push_undo(self, sample_session, sample_facecam_region):
        """Test pushing a region to undo history."""
        new_region = FacecamRegion(
            x=150, y=75, width=350, height=250, corner="top-right", confidence=0.90
        )
        sample_session.push_undo(new_region)
        assert len(sample_session.undo_history) == 1
        assert sample_session.undo_history[0] == new_region

    def test_pop_undo(self, sample_session, sample_facecam_region):
        """Test popping a region from undo history."""
        new_region = FacecamRegion(
            x=150, y=75, width=350, height=250, corner="top-right", confidence=0.90
        )
        sample_session.push_undo(new_region)
        popped = sample_session.pop_undo()
        assert popped == new_region
        assert len(sample_session.undo_history) == 0

    def test_pop_undo_empty(self, sample_session):
        """Test popping from empty undo history returns None."""
        popped = sample_session.pop_undo()
        assert popped is None

    def test_push_redo(self, sample_session, sample_facecam_region):
        """Test pushing a region to redo history."""
        new_region = FacecamRegion(
            x=150, y=75, width=350, height=250, corner="top-right", confidence=0.90
        )
        sample_session.push_redo(new_region)
        assert len(sample_session.redo_history) == 1
        assert sample_session.redo_history[0] == new_region

    def test_pop_redo(self, sample_session, sample_facecam_region):
        """Test popping a region from redo history."""
        new_region = FacecamRegion(
            x=150, y=75, width=350, height=250, corner="top-right", confidence=0.90
        )
        sample_session.push_redo(new_region)
        popped = sample_session.pop_redo()
        assert popped == new_region
        assert len(sample_session.redo_history) == 0

    def test_pop_redo_empty(self, sample_session):
        """Test popping from empty redo history returns None."""
        popped = sample_session.pop_redo()
        assert popped is None

    def test_push_undo_clears_redo(self, sample_session, sample_facecam_region):
        """Test that pushing undo clears redo history."""
        redo_region = FacecamRegion(
            x=200, y=100, width=300, height=200, corner="bottom-left", confidence=0.75
        )
        sample_session.push_redo(redo_region)
        assert len(sample_session.redo_history) == 1

        undo_region = FacecamRegion(
            x=150, y=75, width=350, height=250, corner="top-right", confidence=0.90
        )
        sample_session.push_undo(undo_region)
        assert len(sample_session.undo_history) == 1
        assert len(sample_session.redo_history) == 0  # Cleared

    def test_clear_history(self, sample_session, sample_facecam_region):
        """Test clearing both undo and redo history."""
        region1 = FacecamRegion(
            x=150, y=75, width=350, height=250, corner="top-right", confidence=0.90
        )
        region2 = FacecamRegion(
            x=200, y=100, width=300, height=200, corner="bottom-left", confidence=0.75
        )
        sample_session.push_undo(region1)
        sample_session.push_redo(region2)
        assert len(sample_session.undo_history) == 1
        assert len(sample_session.redo_history) == 1

        sample_session.clear_history()
        assert len(sample_session.undo_history) == 0
        assert len(sample_session.redo_history) == 0


class TestVerticalFormattingJob:
    """Tests for VerticalFormattingJob dataclass."""

    @pytest.fixture
    def sample_facecam_region(self) -> FacecamRegion:
        """Create a sample facecam region."""
        return FacecamRegion(
            x=100,
            y=50,
            width=400,
            height=300,
            corner="top-left",
            confidence=0.85,
        )

    @pytest.fixture
    def sample_canvas_layout(self) -> CanvasLayout:
        """Create a sample canvas layout."""
        return CanvasLayout(
            canvas_width=1080,
            canvas_height=1920,
            facecam_x=0,
            facecam_y=0,
            facecam_width=1080,
            facecam_height=672,
            gameplay_x=0,
            gameplay_y=672,
            gameplay_width=1080,
            gameplay_height=1248,
        )

    @pytest.fixture
    def sample_clips(self) -> list[dict]:
        """Create sample clips list."""
        return [
            {"path": "/path/to/clip1.mp4", "name": "clip1", "resolution": (1920, 1080)},
            {"path": "/path/to/clip2.mp4", "name": "clip2", "resolution": (1920, 1080)},
            {"path": "/path/to/clip3.mp4", "name": "clip3", "resolution": (1920, 1080)},
        ]

    @pytest.fixture
    def sample_job(
        self, sample_facecam_region, sample_canvas_layout, sample_clips
    ) -> VerticalFormattingJob:
        """Create a sample formatting job."""
        return VerticalFormattingJob(
            job_id="job-001",
            session_id="session-001",
            clip_batch_id="batch-001",
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
            settings={"backup": True, "naming": "_vertical"},
            clips=sample_clips,
            output_dir="/output",
        )

    def test_job_creation(self, sample_job, sample_clips):
        """Test that a job is created with correct attributes."""
        assert sample_job.job_id == "job-001"
        assert sample_job.session_id == "session-001"
        assert sample_job.clip_batch_id == "batch-001"
        assert sample_job.status == "queued"
        assert sample_job.clips_processed == 0
        assert sample_job.clips_total == len(sample_clips)
        assert sample_job.current_clip == ""
        assert sample_job.errors == []

    def test_job_start_processing(self, sample_job):
        """Test starting job processing."""
        before = time.time()
        sample_job.start_processing()
        after = time.time()
        assert sample_job.status == "running"
        assert before <= sample_job.started_at <= after

    def test_job_complete_processing(self, sample_job):
        """Test completing job processing."""
        sample_job.start_processing()
        before = time.time()
        sample_job.complete_processing()
        after = time.time()
        assert sample_job.status == "done"
        assert before <= sample_job.completed_at <= after

    def test_job_fail_processing(self, sample_job):
        """Test failing job processing."""
        sample_job.start_processing()
        before = time.time()
        sample_job.fail_processing()
        after = time.time()
        assert sample_job.status == "failed"
        assert before <= sample_job.completed_at <= after

    def test_job_cancel_processing(self, sample_job):
        """Test cancelling job processing."""
        sample_job.start_processing()
        before = time.time()
        sample_job.cancel_processing()
        after = time.time()
        assert sample_job.status == "cancelled"
        assert before <= sample_job.completed_at <= after

    def test_increment_progress(self, sample_job):
        """Test incrementing progress counter."""
        assert sample_job.clips_processed == 0
        sample_job.increment_progress("clip1")
        assert sample_job.clips_processed == 1
        assert sample_job.current_clip == "clip1"
        sample_job.increment_progress("clip2")
        assert sample_job.clips_processed == 2
        assert sample_job.current_clip == "clip2"

    def test_add_error(self, sample_job):
        """Test adding error messages."""
        assert sample_job.errors == []
        sample_job.add_error("Error processing clip1")
        assert len(sample_job.errors) == 1
        assert sample_job.errors[0] == "Error processing clip1"
        sample_job.add_error("Error processing clip2")
        assert len(sample_job.errors) == 2

    def test_get_progress_percentage(self, sample_job):
        """Test progress percentage calculation."""
        assert sample_job.get_progress_percentage() == 0.0
        sample_job.increment_progress()
        assert sample_job.get_progress_percentage() == pytest.approx(33.33, rel=0.01)
        sample_job.increment_progress()
        assert sample_job.get_progress_percentage() == pytest.approx(66.67, rel=0.01)
        sample_job.increment_progress()
        assert sample_job.get_progress_percentage() == 100.0

    def test_get_progress_percentage_zero_total(self):
        """Test progress percentage with zero total clips."""
        job = VerticalFormattingJob(
            job_id="job-001",
            session_id="session-001",
            clip_batch_id="batch-001",
            facecam_region=FacecamRegion(0, 0, 100, 100, "top-left", 0.5),
            canvas_layout=CanvasLayout(1080, 1920, 0, 0, 1080, 672, 0, 672, 1080, 1248),
            settings={},
            clips=[],
            output_dir="/output",
        )
        assert job.get_progress_percentage() == 0.0

    def test_get_elapsed_time_not_started(self, sample_job):
        """Test elapsed time when job hasn't started."""
        assert sample_job.get_elapsed_time() == 0.0

    def test_get_elapsed_time_running(self, sample_job):
        """Test elapsed time while job is running."""
        sample_job.start_processing()
        time.sleep(0.1)
        elapsed = sample_job.get_elapsed_time()
        assert elapsed >= 0.1

    def test_get_elapsed_time_completed(self, sample_job):
        """Test elapsed time after job completes."""
        sample_job.start_processing()
        time.sleep(0.1)
        sample_job.complete_processing()
        elapsed = sample_job.get_elapsed_time()
        assert elapsed >= 0.1

    def test_estimate_remaining_time_not_started(self, sample_job):
        """Test estimated remaining time when job hasn't started."""
        assert sample_job.estimate_remaining_time() == 0.0

    def test_estimate_remaining_time_no_progress(self, sample_job):
        """Test estimated remaining time with no progress."""
        sample_job.start_processing()
        assert sample_job.estimate_remaining_time() == 0.0

    def test_estimate_remaining_time_with_progress(self, sample_job):
        """Test estimated remaining time with progress."""
        sample_job.start_processing()
        time.sleep(0.1)
        sample_job.increment_progress()  # 1 of 3 clips done
        remaining = sample_job.estimate_remaining_time()
        # Should estimate ~0.2 seconds for 2 remaining clips
        assert remaining > 0.0


class TestSessionStore:
    """Tests for SessionStore class."""

    @pytest.fixture
    def session_store(self) -> SessionStore:
        """Create a session store."""
        return SessionStore()

    @pytest.fixture
    def sample_facecam_region(self) -> FacecamRegion:
        """Create a sample facecam region."""
        return FacecamRegion(
            x=100,
            y=50,
            width=400,
            height=300,
            corner="top-left",
            confidence=0.85,
        )

    @pytest.fixture
    def sample_canvas_layout(self) -> CanvasLayout:
        """Create a sample canvas layout."""
        return CanvasLayout(
            canvas_width=1080,
            canvas_height=1920,
            facecam_x=0,
            facecam_y=0,
            facecam_width=1080,
            facecam_height=672,
            gameplay_x=0,
            gameplay_y=672,
            gameplay_width=1080,
            gameplay_height=1248,
        )

    def test_create_session(
        self, session_store, sample_facecam_region, sample_canvas_layout
    ):
        """Test creating a new session."""
        session = session_store.create_session(
            clip_batch_id="batch-001",
            reference_clip_path="/path/to/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        assert session.session_id is not None
        assert session.clip_batch_id == "batch-001"
        assert session.reference_clip_path == "/path/to/clip.mp4"

    def test_get_session(
        self, session_store, sample_facecam_region, sample_canvas_layout
    ):
        """Test retrieving a session."""
        created_session = session_store.create_session(
            clip_batch_id="batch-001",
            reference_clip_path="/path/to/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        retrieved_session = session_store.get_session(created_session.session_id)
        assert retrieved_session is not None
        assert retrieved_session.session_id == created_session.session_id

    def test_get_session_not_found(self, session_store):
        """Test retrieving a non-existent session."""
        session = session_store.get_session("non-existent-id")
        assert session is None

    def test_get_session_expired(
        self, session_store, sample_facecam_region, sample_canvas_layout
    ):
        """Test retrieving an expired session returns None and removes it."""
        created_session = session_store.create_session(
            clip_batch_id="batch-001",
            reference_clip_path="/path/to/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        # Expire the session
        created_session.expires_at = time.time() - 100
        retrieved_session = session_store.get_session(created_session.session_id)
        assert retrieved_session is None
        # Verify it was removed
        assert created_session.session_id not in session_store._sessions

    def test_delete_session(
        self, session_store, sample_facecam_region, sample_canvas_layout
    ):
        """Test deleting a session."""
        created_session = session_store.create_session(
            clip_batch_id="batch-001",
            reference_clip_path="/path/to/clip.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        assert session_store.delete_session(created_session.session_id)
        assert session_store.get_session(created_session.session_id) is None

    def test_delete_session_not_found(self, session_store):
        """Test deleting a non-existent session."""
        assert not session_store.delete_session("non-existent-id")

    def test_cleanup_expired_sessions(
        self, session_store, sample_facecam_region, sample_canvas_layout
    ):
        """Test cleaning up expired sessions."""
        # Create multiple sessions
        session1 = session_store.create_session(
            clip_batch_id="batch-001",
            reference_clip_path="/path/to/clip1.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        session2 = session_store.create_session(
            clip_batch_id="batch-002",
            reference_clip_path="/path/to/clip2.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        # Expire one session
        session1.expires_at = time.time() - 100
        # Cleanup
        removed_count = session_store.cleanup_expired_sessions()
        assert removed_count == 1
        assert session_store.get_session(session1.session_id) is None
        assert session_store.get_session(session2.session_id) is not None

    def test_list_sessions(
        self, session_store, sample_facecam_region, sample_canvas_layout
    ):
        """Test listing all active sessions."""
        session1 = session_store.create_session(
            clip_batch_id="batch-001",
            reference_clip_path="/path/to/clip1.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        session2 = session_store.create_session(
            clip_batch_id="batch-002",
            reference_clip_path="/path/to/clip2.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        sessions = session_store.list_sessions()
        assert len(sessions) == 2
        session_ids = {s.session_id for s in sessions}
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

    def test_list_sessions_excludes_expired(
        self, session_store, sample_facecam_region, sample_canvas_layout
    ):
        """Test that list_sessions excludes expired sessions."""
        session1 = session_store.create_session(
            clip_batch_id="batch-001",
            reference_clip_path="/path/to/clip1.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        session2 = session_store.create_session(
            clip_batch_id="batch-002",
            reference_clip_path="/path/to/clip2.mp4",
            reference_resolution=(1920, 1080),
            facecam_region=sample_facecam_region,
            canvas_layout=sample_canvas_layout,
        )
        # Expire one session
        session1.expires_at = time.time() - 100
        sessions = session_store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == session2.session_id
