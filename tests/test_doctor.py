from meeting_minutes.doctor import doctor_exit_code, render_doctor_report


def test_doctor_report_lists_required_result():
    checks = [
        {
            "name": "python",
            "kind": "runtime",
            "required": True,
            "status": "ok",
            "version": "3.13.0",
            "note": "Python >= 3.11 is required.",
        }
    ]

    report = render_doctor_report(checks)

    assert "[OK] python (required)" in report
    assert "Required checks passed" in report


def test_doctor_strict_exit_code_fails_on_missing_required():
    checks = [{"name": "mlx_whisper", "required": True, "status": "missing"}]

    assert doctor_exit_code(checks, strict=True) == 1
    assert doctor_exit_code(checks, strict=False) == 0
