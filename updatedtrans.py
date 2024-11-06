import csv

def parse_transaction_line(line, start_pos, end_pos):
    """Extracts the transaction ID or amount from a line based on specified positions."""
    return line[start_pos-1:end_pos].strip()  # Subtract 1 for zero-based indexing


def read_csv_transactions(csv_file_path, transaction_column_name):
    """Reads transactions from the CSV file based on the specified column header."""
    transactions_to_exclude = set()
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            transactions_to_exclude.add(row[transaction_column_name].strip())
    return transactions_to_exclude


def update_transaction_file(text_file_path, csv_transactions, transaction_start, transaction_end,
                            amount_start, amount_end, trailer_start, trailer_end):
    """Reads the transaction file, excludes specified transactions, updates the trailer amount, and logs details."""
    updated_lines = []
    total_amount = 0
    initial_total_amount = 0
    skipped_transactions = []

    with open(text_file_path, 'r') as file:
        lines = file.readlines()

        # Process each line
        for line in lines:
            if line.startswith("10 "):  # Identify transaction line by prefix "10"
                transaction_id = parse_transaction_line(line, transaction_start, transaction_end)
                amount_str = parse_transaction_line(line, amount_start, amount_end)

                # Check if amount is valid
                if amount_str:
                    try:
                        amount = int(amount_str)
                        initial_total_amount += amount
                        
                        # Check if transaction ID is in CSV exclusion list
                        if transaction_id not in csv_transactions:
                            total_amount += amount
                            updated_lines.append(line)  # Keep this line
                        else:
                            skipped_transactions.append((transaction_id, amount))  # Log skipped transaction
                    except ValueError:
                        print(f"Skipping line with invalid amount: {line.strip()}")
                        continue  # Skip this line if amount conversion fails
            elif line.startswith("90 "):  # Trailer line
                # Update trailer with new total amount
                formatted_total = f"{total_amount:010}"
                updated_trailer_line = f"{line[:trailer_start-1]}{formatted_total}{line[trailer_end:]}"
                updated_lines.append(updated_trailer_line)
            else:
                # Keep other lines such as header without modification
                updated_lines.append(line)

    # Write the updated content back to a new file
    updated_file_path = text_file_path.replace(".txt", "_updated.txt")
    with open(updated_file_path, 'w') as updated_file:
        updated_file.writelines(updated_lines)

    # Write the control file with details of the process
    control_file_path = text_file_path.replace(".txt", "_control.txt")
    skipped_total_amount = sum(amount for _, amount in skipped_transactions)
    final_total_in_dollars = total_amount / 100  # Convert cents to dollars

    with open(control_file_path, 'w') as control_file:
        control_file.write("Control File for Transaction Processing\n")
        control_file.write("========================================\n\n")
        control_file.write(f"Initial Total Amount (in cents): {initial_total_amount}\n")
        control_file.write(f"Initial Total Amount (in dollars): ${initial_total_amount / 100:.2f}\n\n")
        
        control_file.write("Skipped Transactions:\n")
        control_file.write("----------------------\n")
        for transaction_id, amount in skipped_transactions:
            control_file.write(f"Transaction ID: {transaction_id}, Amount: {amount} cents (${amount / 100:.2f})\n")
        
        control_file.write("\nSkipped Total Amount (in cents): {skipped_total_amount}\n")
        control_file.write(f"Skipped Total Amount (in dollars): ${skipped_total_amount / 100:.2f}\n\n")

        control_file.write("Final Calculations:\n")
        control_file.write("--------------------\n")
        control_file.write(f"Final Total Amount (in cents): {total_amount}\n")
        control_file.write(f"Final Total Amount (in dollars): ${final_total_in_dollars:.2f}\n")
        control_file.write(f"Calculation: Final Total = Initial Total - Skipped Total\n")
        control_file.write(f"              {initial_total_amount} - {skipped_total_amount} = {total_amount}\n")

    print(f"Updated transaction file created at: {updated_file_path}")
    print(f"Control file created at: {control_file_path}")


# Configuration settings based on column headers and positions
csv_file_path = 'sample_transactions.csv'
text_file_path = 'transaction_file.txt'

# Specify the CSV and text file format details for generic parsing
csv_transaction_column_name = "Transaction Number"  # CSV column name for transaction ID to exclude
transaction_start_pos = 16  # Start position of transaction ID in transaction file
transaction_end_pos = 30    # End position of transaction ID in transaction file
amount_start_pos = 47       # Start position of amount in transaction file
amount_end_pos = 56         # End position of amount in transaction file
trailer_amount_start = 21   # Start position of trailer amount in transaction file
trailer_amount_end = 30     # End position of trailer amount in transaction file

# Process
csv_transactions = read_csv_transactions(csv_file_path, csv_transaction_column_name)
update_transaction_file(text_file_path, csv_transactions, transaction_start_pos, transaction_end_pos,
                        amount_start_pos, amount_end_pos, trailer_amount_start, trailer_amount_end)
