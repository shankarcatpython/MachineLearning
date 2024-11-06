from faker import Faker
import random
from datetime import datetime
import csv

# Initialize Faker
fake = Faker()

# File settings
num_transactions = 100  # Generate 100 records
sample_percentage = 0.10  # 10% of transactions for CSV file
file_path = 'transaction_file.txt'
csv_file_path = 'sample_transactions.csv'

# Generate data
transaction_ids = []  # Store transaction IDs for sampling in CSV file

with open(file_path, 'w') as file:
    # Header with identifier, timestamp, and data section
    file_creation_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header_line = f"10 HEADER {file_creation_timestamp} 20\n"
    file.write(header_line)
    
    # Transactions
    total_amount = 0
    for _ in range(num_transactions):
        # Generate date
        date = fake.date_this_year().strftime('%Y-%m-%d')
        
        # Generate a fixed 15-byte alphanumeric transaction ID
        transaction_id = fake.bothify(text='?' * 15, letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        transaction_ids.append(transaction_id)  # Collect for CSV sampling
        
        # Generate location and format it to a fixed width of 15 characters, right-padded with spaces
        location = fake.city()[:15].ljust(15)  # Trim to 15 characters if necessary and pad with spaces
        
        # Generate amount (in cents) and format it as a 10-byte integer with leading zeros
        amount_in_cents = random.randint(100, 99999999)  # Random amount from 1.00 to 999999.99
        formatted_amount = f"{amount_in_cents:010}"  # 10 bytes, leading zeros
        
        # Update total amount
        total_amount += amount_in_cents

        # Format and write the transaction line, beginning with '10', followed by date
        line = f"10 {date}  {transaction_id} {location}{formatted_amount}\n"
        file.write(line)
    
    # Trailer with identifier and total amount (formatted as 10-byte integer with leading zeros)
    formatted_total = f"{total_amount:010}"
    trailer_line = f"90 TRAILER {formatted_total}\n"
    file.write(trailer_line)

# Generate a CSV file with 10% sample of transaction numbers
sample_size = int(num_transactions * sample_percentage)
sample_transaction_ids = random.sample(transaction_ids, sample_size)  # Randomly select 10% of transaction IDs

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['S.No', 'Name', 'Transaction Number'])  # Write header
    
    for i, transaction_id in enumerate(sample_transaction_ids, start=1):
        name = fake.name()  # Generate a random name
        csv_writer.writerow([i, name, transaction_id])  # Write row

print(f"Transaction file with 100 records created at: {file_path}")
print(f"CSV file with sample transactions created at: {csv_file_path}")
