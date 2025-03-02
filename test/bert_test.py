import pytest

from wwt_stuff.phishing_detector.bert_model.bert_run import (
    PredictionResult,
    predict_email,
)


@pytest.mark.parametrize(
    ("email_subject", "email_body", "expected"),
    [
        (
            "Update: New Polling Locations Announced",
            "QUICK, you've fallen behind on your student loans. Click here to pay them at a discounted rate",
            "Phishing Email",
        ),
        (
            "Lunch",
            "Let's go get lunch, im hungry",
            "Legitimate Email",
        ),
        pytest.param(
            "Chrome Update",
            """This Friday MIS will push version 108.0.5359.94/96 of the Chrome browser to your device.

Once the latest Chrome update reaches your device, a notification will appear giving you the option to either update the browser immediately or snooze for up to 8 hours. To update, simply accept the notification, quit your browser, and then relaunch it approximately 10 minutes later to allow the install to complete.

After successful installation,
click here to disable location tracking (per security policy).

If you encounter any issues with Chrome, please submit a Support Request for further assistance.

Thank you!

- The MIS Team""",
            "Phishing Email",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            "New Polling Locations Announced",
            """As part of our ongoing efforts to ensure a smooth and accessible voting process, we have updated the list of polling locations for the upcoming election. Please review the updated polling locations to find your nearest voting center.

Click here to find Your Polling Location.

Staying informed about your designated polling location helps us ensure a secure and efficient voting experience for everyone.

Thank you for your attention to this important matter.

Sincerely,
USA.gov Election Assistance Team
info@usa.gov""",
            "Phishing Email",
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_bert(email_subject: str, email_body: str, expected: PredictionResult) -> None:
    assert predict_email(email_subject, email_body).result == expected
