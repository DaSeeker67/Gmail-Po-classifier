# Gmail-Po-classifier
Gmail Po classifier
Purchase Order Email Processor

Overview

This project is a Streamlit-based application for processing purchase order emails. It extracts structured details such as customer information, item details, and total amounts from email bodies and attachments. The app uses Natural Language Processing (NLP) and OCR techniques to process text from emails and attachments like PDFs, images, and Excel files.

Setup Instructions

Prerequisites

Ensure you have the following installed on your system:

Python (>= 3.8)

Tesseract OCR

Linux: Install via sudo apt-get install tesseract-ocr

Windows: Download and install from Tesseract OCR

MacOS: Install via Homebrew: brew install tesseract

Internet connection for downloading pre-trained models and packages.

Step-by-Step Setup

Clone the Repository:

git clone <repository-url>
cd <repository-folder>

Create a Virtual Environment:

python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Run the Application:

streamlit run app.py

Environment Variables:
Ensure you have access credentials for your email server (IMAP) and update the app with them during runtime.

Approach

Key Features

Email Classification: Classifies emails into "Purchase Order (PO)" or "Non-PO" using a machine learning model.

Attachment Processing: Extracts text content from image, PDF, and Excel attachments.

Structured Extraction: Extracts customer and item details using entity recognition.

User Interaction: Allows manual verification and editing of extracted PO details via the Streamlit interface.

Technical Details

Email Processing:

Connects to an IMAP server to fetch unread emails.

Decodes email content and processes attachments in memory.

NLP Classification:

A custom Random Forest Classifier was trained to classify emails as "PO" or "Non-PO."

If "purchase order" is explicitly mentioned in the email subject, it is automatically classified as a "PO."

Entity Extraction:

Used GLiNER for recognizing key entities such as customer name, phone number, email, item details, and amounts.

OCR for Attachments:

Utilized PyTesseract for extracting text from image attachments.

Used PyPDF2 and pandas for processing PDFs and Excel files respectively.

Key Challenges Faced

Lack of Data for PO Classification:

Training data was not readily available, so we created our own dataset using the Enron dataset.

Generated a balanced dataset of 1,000 Purchase Orders (PO) and 1,000 Non-PO emails.

Model Performance:

Initial experiments with fine-tuned BERT models did not yield satisfactory results due to the limited size of our dataset.

Simpler models like Random Forest performed better, achieving satisfactory accuracy and generalizability.

Next Steps

Expand the dataset to include more examples for better generalization.

Incorporate additional attachment formats such as .csv and .txt.

Improve entity extraction by fine-tuning a transformer-based model on our dataset.

Contributing

Feel free to fork the repository and create pull requests for any improvements or bug fixes.
