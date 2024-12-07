import streamlit as st
import imaplib
import email
import pandas as pd
import re
import os
import string
from email.header import decode_header
from datetime import datetime
from gliner import GLiNER
import joblib
import pypdf
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import pandas as pd
from io import BytesIO

class EmailPOProcessor:
    def __init__(self, imap_server, email_address, password):
        """
        Initialize email processor with IMAP credentials
        """
        self.imap_server = imap_server
        self.email_address = email_address
        self.password = password
        self.model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        
        # Add stop words import or definition
        self.stop_words = set(['the', 'a', 'an', 'in', 'to', 'for', 'of', 'and', 'or', 'but'])

    def decode_email_body(self, payload):
        """
        Decode email body with fallback mechanisms
        """
        try:
            return payload.decode("utf-8", errors="replace")
        except Exception:
            return "Unable to decode email body"

    def connect_to_mailbox(self):
        """
        Establish IMAP connection to mailbox
        """
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.email_address, self.password)
            mail.select('inbox')
            return mail
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            return None

    def preprocess_text(self, text):
        """
        Preprocess text for classification
        """
        if text is None:
            return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)

    def extract_po_details(self, email_body: str,attachements: str) -> dict:
        """
        Extract structured PO details using GLiNER for entity recognition
        """
        labels = [
            "Customer PO Number", "Item Name", "Quantity", "Rate per unit",
            "Unit of measurement", "Customer Name", "phone number", "email",
            "full address", "Total Amount"
        ]
       
        extracted_entities = self.model.predict_entities(email_body+attachements, labels)
       
        # Create a dictionary to store extracted information
        po_details = {
            "Customer PO Number": "N/A",
            "Item Details": [],
            "Customer Name": "N/A",
            "Customer Phone": "N/A",
            "Customer Email": "N/A",
            "Delivery Address": "N/A",
            "Total Amount": "N/A"
        }
       
        # Map the actual extracted entities to our po_details
        for entity in extracted_entities:
            text = entity['text']
            label = entity['label']
       
            if label == "Customer Name":
                po_details["Customer Name"] = text
            elif label == "phone number":
                po_details["Customer Phone"] = text
            elif label == "email":
                po_details["Customer Email"] = text
            elif label == "Customer PO Number":
                po_details["Customer PO Number"] = text
            elif label == "full address":
                po_details["Delivery Address"] = text
            elif label == "Total Amount":
                po_details["Total Amount"] = text
            elif label == "Item Name":
                # Create an item details dictionary
                item_detail = {
                    "Item Name": text,
                    "Quantity": "N/A",
                    "Rate per unit": "N/A",
                    "Unit of measurement": "N/A"
                }
                
                # Try to find corresponding details for this item
                for additional_entity in extracted_entities:
                    if additional_entity['start'] > entity['end']:
                        if additional_entity['label'] == "Quantity":
                            item_detail["Quantity"] = additional_entity['text']
                        elif additional_entity['label'] == "Rate per unit":
                            item_detail["Rate per unit"] = additional_entity['text']
                        elif additional_entity['label'] == "Unit of measurement":
                            item_detail["Unit of measurement"] = additional_entity['text']
                
                po_details["Item Details"].append(item_detail)
       
        return po_details
    
        
    def process_email_attachments(self, msg) -> list:
        """
        Process email attachments and extract text content from images, PDFs, and Excel files.
        """
        extracted_content = []
    
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    file_data = part.get_payload(decode=True)
                    
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        # Extract text from image
                        image_text = pytesseract.image_to_string(file_data)
                        extracted_content.append({'filename': filename, 'content': image_text})
                    
                    elif filename.lower().endswith('.pdf'):
                        # Extract text from PDF
                        pdf_text = ""
                        try:
                            pdf_reader = PdfReader(BytesIO(file_data))
                            pdf_text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                        except Exception as e:
                            print(f"Error processing PDF {filename}: {e}")
                        extracted_content.append({'filename': filename, 'content': pdf_text})
                    
                    elif filename.lower().endswith(('.xls', '.xlsx')):
                        # Extract text from Excel
                        try:
                            excel_data = pd.read_excel(BytesIO(file_data), sheet_name=None)
                            excel_text = "\n".join(
                                f"Sheet: {sheet_name}\n{sheet_df.to_string(index=False)}"
                                for sheet_name, sheet_df in excel_data.items()
                            )
                        except Exception as e:
                            print(f"Error processing Excel {filename}: {e}")
                            excel_text = ""
                        extracted_content.append({'filename': filename, 'content': excel_text})
        content = " ".join(x['content'] for x in extracted_content)
        return content

    def classify_email(self, subject, body):
        """
        Classify email as PO or Non-PO
        """
        if "purchase order" in subject.lower():
           return "PO"
        model = joblib.load("po_classifier_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        combined_text = subject + " " + body
        preprocessed_text = self.preprocess_text(combined_text)
        text_vectorized = vectorizer.transform([preprocessed_text])
        prediction = model.predict(text_vectorized)
        return "PO" if prediction[0] == 1 else "Non-PO"

    def fetch_most_recent_unread_emails(self, max_emails=30):
        """
        Fetch most recent unread emails
        """
        mail = self.connect_to_mailbox()
        if not mail:
            return []

        _, search_data = mail.search(None, 'UNSEEN')
        email_ids = search_data[0].split()
        email_ids = sorted(email_ids, key=lambda x: int(x), reverse=True)[:max_emails]

        processed_emails = []
        for num in email_ids:
            _, data = mail.fetch(num, '(RFC822)')
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            email_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        email_body = self.decode_email_body(payload)
                        break
            else:
                payload = msg.get_payload(decode=True)
                email_body = self.decode_email_body(payload)

            email_date = email.utils.parsedate_to_datetime(msg['Date']) if msg['Date'] else datetime.now()
            
            # Classify the email first
            email_classification = self.classify_email(msg['subject'], email_body)
            attachments = self.process_email_attachments(msg)
            po_details = self.extract_po_details(email_body,attachments) if email_classification == "PO" else None
            

            processed_emails.append({
                'date': email_date,
                'subject': msg['subject'],
                'from': msg['from'],
                'body': email_body,
                'po_details': po_details,
                'attachments': attachments,
                'classification': email_classification
            })

        processed_emails.sort(key=lambda x: x['date'], reverse=True)
        return processed_emails

# Rest of the code remains the same as in the previous submission
def display_po_details(email_data, email_index):
    """
    Display and allow editing of PO details
    """
    # If email is not a Purchase Order, display a simple message
    if email_data['classification'] == "Non-PO":
        st.warning("This is not a Purchase Order email.")
        
        # Display basic email details
        st.markdown(f"### Email Details")
        st.write(f"**From:** {email_data['from']}")
        st.write(f"**Subject:** {email_data['subject']}")
        st.write(f"**Date:** {email_data['date']}")

        with st.expander("Original Email Body"):
            st.text(email_data['body'])
        return None

    # Rest of the existing implementation remains the same for PO emails
    po_details = email_data['po_details']

    st.markdown(f"### Email Details")
    st.write(f"**From:** {email_data['from']}")
    st.write(f"**Subject:** {email_data['subject']}")
    st.write(f"**Date:** {email_data['date']}")

    with st.expander("Original Email Body"):
        st.text(email_data['body'])

    st.markdown("### Purchase Order Details")
    
    # Create form for manual input and verification
    with st.form(key=f'po_details_form_{email_index}', clear_on_submit=False):  # Use unique key
        # Basic PO Information
        col1, col2 = st.columns(2)
        with col1:
            po_number = st.text_input(
                "Purchase Order Number", 
                value=po_details.get('Customer PO Number', 'N/A')
            )
            customer_name = st.text_input(
                "Customer Name", 
                value=po_details.get('Customer Name', 'N/A')
            )
        with col2:
            customer_phone = st.text_input(
                "Customer Phone", 
                value=po_details.get('Customer Phone', 'N/A')
            )
            customer_email = st.text_input(
                "Customer Email", 
                value=po_details.get('Customer Email', 'N/A')
            )

        # Delivery Information
        delivery_address = st.text_input(
            "Delivery Address", 
            value=po_details.get('Delivery Address', 'N/A')
        )
        total_amount = st.text_input(
            "Total Amount", 
            value=po_details.get('Total Amount', 'N/A')
        )

        # Item Details
        st.markdown("### Item Details")
        
        # Track number of items using session state
        if f'item_count_{email_index}' not in st.session_state:
            st.session_state[f'item_count_{email_index}'] = len(po_details.get('Item Details', []))

        # Dynamically generate item input fields
        item_details = []
        for i in range(st.session_state[f'item_count_{email_index}']):
            with st.expander(f"Item {i+1}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    item_name = st.text_input(
                        "Item Name", 
                        value=po_details['Item Details'][i]['Item Name'] if i < len(po_details.get('Item Details', [])) else '',
                        key=f"item_name_{email_index}_{i}"
                    )
                with col2:
                    quantity = st.text_input(
                        "Quantity", 
                        value=po_details['Item Details'][i]['Quantity'] if i < len(po_details.get('Item Details', [])) else '',
                        key=f"quantity_{email_index}_{i}"
                    )
                with col3:
                    rate = st.text_input(
                        "Rate per Unit", 
                        value=po_details['Item Details'][i]['Rate per unit'] if i < len(po_details.get('Item Details', [])) else '',
                        key=f"rate_{email_index}_{i}"
                    )
                with col4:
                    unit = st.text_input(
                        "Unit of Measurement", 
                        value=po_details['Item Details'][i]['Unit of measurement'] if i < len(po_details.get('Item Details', [])) else '',
                        key=f"unit_{email_index}_{i}"
                    )
                
                item_details.append({
                    "Item Name": item_name,
                    "Quantity": quantity,
                    "Rate per unit": rate,
                    "Unit of measurement": unit
                })

        # Add Another Item column within the form
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("")  # Placeholder to align the button
        with col2:
            add_item = st.form_submit_button("+ Item")
            if add_item:
                st.session_state[f'item_count_{email_index}'] += 1
                st.experimental_rerun()

        # Submit button
        submit_button = st.form_submit_button("Save Purchase Order Details")
        
        if submit_button:
            # Create updated PO details dictionary
            updated_po_details = {
                "Customer PO Number": po_number,
                "Customer Name": customer_name,
                "Customer Phone": customer_phone,
                "Customer Email": customer_email,
                "Delivery Address": delivery_address,
                "Total Amount": total_amount,
                "Item Details": item_details
            }
            
            # You can add further processing here, like saving to a database
            st.success("Purchase Order Details Saved Successfully!")
            return updated_po_details

def main():
    st.title('Purchase Order Email Processor')

    # Sidebar for Email Configuration
    st.sidebar.header('Email Configuration')
    imap_server = st.sidebar.text_input('IMAP Server', 'imap.gmail.com')
    email_address = st.sidebar.text_input('Email Address')
    password = st.sidebar.text_input('Password', type='password')

    if st.sidebar.button('Fetch Most Recent Unread Emails'):
        processor = EmailPOProcessor(imap_server, email_address, password)
        emails = processor.fetch_most_recent_unread_emails()

        st.write(f"Total Recent Unread Emails Processed: {len(emails)}")

        # Tabs for different emails
        email_tabs = st.tabs([f"Email {i+1}" for i in range(len(emails))])
        
        for i, (email_tab, email_data) in enumerate(zip(email_tabs, emails)):
            with email_tab:
                display_po_details(email_data, email_index=i)

if __name__ == '__main__':
    main()