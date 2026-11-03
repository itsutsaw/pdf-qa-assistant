"""Build examples/eval_handbook.pdf, a fictional multi-page handbook for evaluate.py.

Each section is one PDF page, so a question's expected page is the section's position.
Rerun after editing SECTIONS: python scripts/make_eval_pdf.py (needs requirements-dev.txt).
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUTPUT = Path(__file__).parents[1] / "examples" / "eval_handbook.pdf"

SECTIONS = [
    ("Welcome to Larkspur Analytics", [
        "Larkspur Analytics is a fictional data consultancy with offices in Bengaluru and Pune. "
        "This handbook explains the policies that apply to permanent employees. It is written for "
        "testing document question answering and does not describe any real organisation.",
        "Each policy has an owner team. People Operations owns working time, leave and pay. The "
        "Finance team owns travel and expenses. The Security Office owns information security and "
        "remote work abroad. Facilities owns health and safety at our offices.",
        "Most requests use a short form code, for example HR-12 or TRV-2. All forms are available "
        "on the intranet under Self Service. If a form is unavailable, email the owner team and "
        "describe your request in writing; a written request has the same effect as the form.",
        "Policies are reviewed every January. When a policy changes, People Operations publishes a "
        "summary of the change on the intranet and the new version applies from the first day of "
        "the following month. If this handbook and a signed employment contract disagree, the "
        "contract takes precedence.",
        "New employees complete an induction during their first week. Induction covers the tools we "
        "use, how projects are staffed, and who to contact in each owner team. Your manager will also "
        "assign a buddy from another team who can answer informal questions during your first three "
        "months.",
        "Questions about a policy should go first to your manager. If your manager cannot answer, "
        "contact the owner team listed above. Decisions that make an exception to a policy must be "
        "recorded in writing so that the same request is treated consistently in future.",
    ]),
    ("Working hours and attendance", [
        "Core hours are 10:00 AM to 4:00 PM, Monday to Friday. Outside core hours you may choose "
        "when to start and finish, provided you work 40 hours per week on average over a month.",
        "Record your hours in a weekly timesheet. Timesheets are submitted with form HR-12 by "
        "12:00 noon every Monday for the previous week. Client projects are billed from these "
        "timesheets, so late or missing timesheets delay invoices.",
        "If you are running late or cannot attend, tell your manager before 10:00 AM by message or "
        "phone. Repeated unexplained absence is handled under the conduct policy.",
        "Overtime must be agreed in advance by your manager. Approved overtime is compensated with "
        "time off in lieu at the same number of hours; it is not paid as additional salary. Time off "
        "in lieu must be used within three months of the overtime being worked.",
        "Part-time employees agree their working pattern with their manager when they join. The same "
        "core hours apply on the days they work, and their leave allowance is calculated in "
        "proportion to their contracted hours.",
        "Meetings should be scheduled within core hours wherever possible, so that colleagues using "
        "flexible start and finish times can attend. Recurring meetings outside core hours need the "
        "agreement of everyone invited.",
    ]),
    ("Annual leave and public holidays", [
        "Employees receive 24 days of paid annual leave in each leave year. The leave year runs from "
        "1 January to 31 December. New joiners receive a pro-rated allowance for their first year.",
        "Request annual leave with form HR-17. For absences longer than three days, submit form HR-17 "
        "at least 10 working days before the first day of leave. Shorter absences need two working "
        "days of notice. Leave is confirmed only after your manager approves the request.",
        "You may carry over up to 5 unused days into the next leave year. Carried-over days must be "
        "used by 31 March, after which they expire. Unused leave is not paid out except when you "
        "leave the company.",
        "The company observes 12 public holidays each year. The list is published in December for "
        "the following year. If you work on a public holiday at a client's request, you receive one "
        "replacement day of leave.",
        "Managers may decline a leave request if too many people in the same team have already booked "
        "the same dates, or if a client deadline makes the absence impractical. A declined request "
        "should be discussed with you and an alternative date offered.",
        "If you fall ill during annual leave, the days of illness can be recorded as sick leave "
        "instead, provided you tell your manager at the time and follow the normal sick leave rules.",
    ]),
    ("Sick leave and family leave", [
        "Employees receive 10 days of paid sick leave per leave year. Tell your manager as early as "
        "possible on the first day of illness. A medical certificate is required after three "
        "consecutive days of sick leave. Unused sick leave does not carry over.",
        "Primary caregivers receive 26 weeks of paid parental leave. Secondary caregivers receive 4 "
        "weeks of paid parental leave, which can be taken in up to two blocks within the first year "
        "after a birth or adoption.",
        "Bereavement leave of 5 paid days is available on the death of a close family member. "
        "Additional unpaid leave can be agreed with your manager and People Operations.",
        "When you return from parental leave or from more than two weeks of sick leave, your manager "
        "will arrange a return-to-work meeting to agree a phased schedule if you need one.",
        "Medical appointments that cannot be scheduled outside working hours do not need to be "
        "recorded as sick leave. Agree the time with your manager and make up the hours where "
        "practical.",
        "Employees with caring responsibilities for an elderly or disabled relative may request a "
        "temporary change to their working pattern. People Operations will review each request within "
        "two weeks and confirm the outcome in writing.",
    ]),
    ("Remote and hybrid work", [
        "Employees may work remotely for up to 3 days per week. Teams agree shared office days so "
        "that collaboration time is predictable. Client site visits count as office days.",
        "To help set up a safe workspace, the company pays a one-time home office stipend of "
        "INR 15,000. Claim it within your first three months using the normal expense process. The "
        "stipend is not repeated if you change address.",
        "Working from another country is permitted for a maximum of 30 days in any 12-month period. "
        "Before working abroad you must complete security review SEC-09 with the Security Office, "
        "because tax, data protection and export rules differ between countries.",
        "Remote work uses the same security rules as the office. Work only on a company laptop or an "
        "enrolled personal device, lock your screen when you step away, and avoid public Wi-Fi "
        "without the company VPN.",
        "Your home workspace should have a suitable chair, enough light and a stable internet "
        "connection. Facilities can provide a checklist and advice on setting up a comfortable desk.",
        "Managers may ask for more office days during project kick-offs, client workshops or training "
        "weeks. Such requests should be made at least one week in advance so that you can plan "
        "travel.",
    ]),
    ("Business travel", [
        "All business travel must be approved before booking. Submit form TRV-2 with the purpose, "
        "dates and estimated cost. Your manager approves domestic trips; international trips also "
        "need approval from the Finance team.",
        "Book travel through the company travel desk at least 14 days before departure where "
        "possible. Late bookings are allowed for urgent client work, but the reason must be stated "
        "on form TRV-2.",
        "Flights shorter than 6 hours are booked in economy class. Longer flights may be booked in "
        "premium economy. Trains are preferred over flights for journeys under four hours.",
        "Hotel stays in metro cities are capped at INR 6,000 per night including taxes. Choose hotels "
        "from the preferred list where available. Travel insurance is provided automatically for "
        "approved trips.",
        "Keep your manager informed of your travel plans and share an itinerary with the travel desk. "
        "If plans change during a trip, update the booking through the travel desk rather than "
        "booking directly with an airline or hotel.",
        "Frequent flyer points and hotel loyalty rewards earned on business travel may be kept, but "
        "they must never influence the choice of supplier or the cost of a booking.",
    ]),
    ("Expenses and reimbursement", [
        "Submit expense claims with form EXP-4 within 30 days of the date the cost was incurred. "
        "Claims submitted later than 30 days need written approval from the Finance team.",
        "During approved travel a daily allowance of INR 900 covers meals and incidentals. You do "
        "not need receipts for the daily allowance. For other costs, attach a receipt for every item "
        "above INR 300.",
        "Each claim must include the cost centre of the project or team that will pay for it. If you "
        "are unsure of the code, ask your manager before submitting the claim.",
        "Approved claims are reimbursed in the next payroll cycle. Alcohol, fines, personal "
        "entertainment and upgrades you chose for personal reasons are not reimbursable.",
        "Receipts must show the date, supplier and amount. A card statement alone is not accepted as "
        "a receipt. If a receipt is lost, complete the missing receipt declaration in the expense "
        "system and explain the circumstances.",
        "Finance checks a sample of claims every month. If a claim breaks this policy, the amount may "
        "be rejected or recovered from a later payroll, and repeated breaches are handled under the "
        "conduct policy.",
    ]),
    ("IT equipment and support", [
        "Every employee receives a company laptop. Laptops are refreshed every 36 months, or earlier "
        "if a device fails and cannot be repaired. Return your old laptop within two weeks of "
        "receiving a replacement.",
        "If a laptop, phone or security key is lost or stolen, report it to the IT service desk "
        "within 1 hour of noticing. Call extension 4357 or email servicedesk@example.com. The service "
        "desk will lock the device and reset your credentials.",
        "Personal phones and tablets may access work email only after they are enrolled in mobile "
        "device management. Enrollment lets the service desk wipe work data if the device is lost.",
        "Do not install unapproved software on company laptops. Request new software through the "
        "service desk portal so that the license and security checks can be completed.",
        "Keep your laptop updated by restarting it at least once a week so that security patches can "
        "be installed. Devices that miss updates for 30 days may be blocked from the network until "
        "they are updated.",
        "Accessories such as a monitor, keyboard, mouse and headset can be requested through the "
        "service desk portal. Accessories are company property and are returned when you leave.",
    ]),
    ("Information security", [
        "Passwords must be at least 14 characters long and must not be reused from personal "
        "accounts. Change a password only if you suspect it has been exposed. Never share passwords "
        "with anyone, including coworkers and managers.",
        "Multi-factor authentication is required on all work accounts. Administrators must use a "
        "hardware security key as their second factor.",
        "Information is classified as Public, Internal, Confidential or Restricted. Confidential "
        "information may be shared only with people who need it for their work. Restricted "
        "information must never be copied to USB drives or personal cloud storage.",
        "Report suspicious emails with the Report Phish button in the mail client. If you clicked a "
        "link or entered credentials, also contact the IT service desk immediately.",
        "Lock printed confidential documents away when you leave your desk and shred them when they "
        "are no longer needed. Do not discuss client information in public places where you may be "
        "overheard.",
        "Security incidents, such as sending a file to the wrong person, must be reported to the "
        "Security Office as soon as you notice them. Early reporting limits the impact and is never "
        "treated as a disciplinary matter by itself.",
    ]),
    ("Learning and development", [
        "Each employee has an annual learning budget of INR 40,000 for courses, books and "
        "certifications. Unused budget does not carry over to the next year.",
        "To attend a conference or external course, submit form LD-3 to your manager at least one "
        "month in advance. If the event requires travel, a separate travel request is also needed.",
        "Employees receive 2 paid study days per year to prepare for professional examinations. "
        "Study days are booked like annual leave but do not reduce your leave allowance.",
        "If the company pays more than INR 100,000 for a single course and you resign within 12 "
        "months of completing it, you must repay a share of the cost in proportion to the months "
        "remaining.",
        "Managers discuss learning goals with each employee during performance reviews. Approved "
        "learning activities should link to your current role or to a role you are preparing for.",
        "Internal workshops, mentoring sessions and lunch-and-learn talks are free and do not use "
        "your learning budget. The schedule of internal sessions is published on the intranet every "
        "quarter.",
    ]),
    ("Performance and pay", [
        "Performance reviews take place twice a year, in April and October. Each review covers goals, "
        "feedback from colleagues and a development plan for the next six months.",
        "Salaries are paid on the last working day of each month by bank transfer. Payslips are "
        "available on the payroll portal two days before payment.",
        "Pay is reviewed once a year after the April review. Any change applies from 1 May. Pay bands "
        "for each role are published on the intranet.",
        "A discretionary annual bonus may be awarded based on company results and your review "
        "ratings. Eligibility requires at least six months of service by 31 March.",
        "If you disagree with a review outcome, raise it with your manager first. If the disagreement "
        "remains, you may ask People Operations for a second review by another senior manager within "
        "14 days of receiving the outcome.",
        "Promotions are considered during the October review. A promotion case includes evidence of "
        "work at the next level and feedback from at least three colleagues.",
    ]),
    ("Health, safety and wellbeing", [
        "If the fire alarm sounds, leave the building by the nearest exit and gather at the assembly "
        "point in car park B. Do not use the lifts. Fire wardens will confirm when it is safe to "
        "return.",
        "Each office has trained first aiders whose names are posted near the reception desk. First "
        "aid kits are kept in every kitchen.",
        "Report any accident, injury or near miss within 24 hours using form HS-1, even if nobody was "
        "hurt. Facilities reviews every report to prevent repeat incidents.",
        "The employee assistance programme offers 6 free confidential counselling sessions per year "
        "for employees and their immediate family. Sessions can be booked by phone or video.",
        "Hazards such as loose cables, broken chairs or spills should be reported to Facilities so "
        "that they can be fixed quickly. Do not attempt electrical repairs yourself.",
        "Employees who need adjustments for a health condition or disability can request them from "
        "People Operations. Requests are handled confidentially and adjustments are reviewed once a "
        "year.",
    ]),
    ("Conduct and leaving the company", [
        "Treat colleagues, clients and suppliers with respect. Harassment and discrimination are "
        "never acceptable and can be reported to People Operations or anonymously through the ethics "
        "line.",
        "Gifts or hospitality worth more than INR 2,000 must be declared in the gifts register within "
        "one week of receiving them. Cash gifts must always be refused.",
        "The notice period is 60 days for both the employee and the company, unless your contract "
        "states otherwise. Remaining annual leave may be used during the notice period with manager "
        "approval.",
        "Return your laptop, security key and access card on your last working day. People "
        "Operations will invite you to an optional exit interview in your final week.",
        "Conflicts of interest, such as a family member working for a supplier, must be declared to "
        "your manager and People Operations. Declaring a conflict early allows it to be managed "
        "fairly.",
        "Confidentiality obligations continue after you leave. Do not take client data, source code "
        "or internal documents with you, and delete any work files stored on personal devices.",
    ]),
]


def build(path: Path = OUTPUT):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="Larkspur Analytics Employee Handbook",
                            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    story = []
    for number, (title, paragraphs) in enumerate(SECTIONS, start=1):
        story.append(Paragraph("PAGEWISE / EVALUATION DOCUMENT", styles["Normal"]))
        story.append(Paragraph(title, styles["Heading1"]))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 6))
        story.append(Paragraph(f"Larkspur Employee Handbook | Page {number} of {len(SECTIONS)}",
                               styles["Italic"]))
        if number < len(SECTIONS):
            story.append(PageBreak())
    doc.build(story)


if __name__ == "__main__":
    build()
    print(f"Wrote {OUTPUT}")
