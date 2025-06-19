from app import app, db, LoanRequest

with app.app_context():
    records = LoanRequest.query.all()
    print(f"Number of records: {len(records)}")
    
    if records:
        print("\nSample record:")
        record = records[0]
        print(f"Gender: {record.gender}")
        print(f"Married: {record.married}")
        print(f"Education: {record.education}")
        print(f"Prediction: {record.prediction}")
    else:
        print("\nNo records found in database.") 