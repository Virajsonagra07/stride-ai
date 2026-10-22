"""Build a sample store-policy PDF that the assistant retrieves from (document RAG).
Replace data/docs/*.pdf with the store's real policy / size-guide PDFs.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SECTIONS = [
    ("Returns and Exchanges",
     "Unworn footwear can be exchanged within 7 days of purchase at any of our three stores with the "
     "original bill or the digital return code sent on WhatsApp. Refunds are issued to the original payment "
     "method within 5 working days; cash purchases are refunded as store credit. Footwear bought on sale "
     "or clearance can be exchanged for size only and is not eligible for refund. Worn footwear, footwear "
     "without the box, and customised items cannot be returned."),
    ("Size Exchange",
     "If the size does not fit, we offer a free size exchange within 15 days even if the bill is misplaced, "
     "as long as the purchase can be found in our system using your phone number. If your size is not "
     "available in your store, staff can transfer the pair from another branch within 48 hours."),
    ("Manufacturing Defects and Warranty",
     "Formal and leather footwear carries a 90-day warranty against sole separation and stitching defects. "
     "Sports shoes carry a 60-day warranty. Slippers and sandals under Rs 999 carry a 30-day warranty. "
     "Warranty does not cover normal wear, water damage on leather, or damage from improper use. "
     "Bring the pair and bill to any store; repairs or replacement are decided within 3 days."),
    ("Size Guide",
     "We use Indian/UK sizing. Indian/UK size 6 = EU 40 = US men 7; UK 7 = EU 41 = US men 8; "
     "UK 8 = EU 42 = US men 9; UK 9 = EU 43 = US men 10; UK 10 = EU 44 = US men 11. For women, "
     "UK 4 = EU 37 = US women 6; UK 5 = EU 38 = US women 7; UK 6 = EU 39 = US women 8. "
     "If you are between sizes, choose the larger size for sports shoes and the smaller size for leather "
     "formals, since leather stretches slightly with wear. Kids should have about one thumb-width of space "
     "at the toe."),
    ("Payments",
     "We accept cash, UPI (GPay, PhonePe, Paytm), debit and credit cards. No-cost EMI is available on card "
     "purchases above Rs 3,000. GST invoices are issued for every purchase; ask at billing for a business "
     "GST invoice with your GSTIN."),
    ("Loyalty Programme",
     "Customers earn 1 point per Rs 100 spent, linked to their phone number. 100 points = Rs 100 off on the "
     "next purchase. Points expire after 12 months. Birthday month customers get an additional 10 percent off "
     "on full-price items."),
    ("Care Instructions",
     "Leather: wipe with a dry cloth, apply cream polish every two weeks, and keep away from rain. Suede: use a "
     "suede brush, never wash with water. Sports and canvas shoes: remove insoles and hand wash with mild "
     "detergent, dry in shade, never in direct sun or a washing machine dryer. Store shoes with shoe trees or "
     "newspaper to keep the shape."),
    ("Home Delivery and Reservation",
     "Customers can reserve any in-stock pair on WhatsApp for 24 hours and pick it up at the chosen store. "
     "Free home delivery is available within city limits for orders above Rs 1,499, usually the same day if "
     "ordered before 4 PM."),
]

def main():
    out = Path(__file__).parent / "docs" / "store_policies.pdf"
    out.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph("Customer Policies and Size Guide (Sample)", styles["Title"]), Spacer(1, 12)]
    for title, body in SECTIONS:
        story += [Paragraph(title, styles["Heading2"]), Paragraph(body, styles["BodyText"]), Spacer(1, 10)]
    SimpleDocTemplate(str(out), pagesize=A4).build(story)
    print("wrote", out)

if __name__ == "__main__":
    main()
