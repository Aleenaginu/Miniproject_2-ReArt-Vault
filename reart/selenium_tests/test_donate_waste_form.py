from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Path to your WebDriver (update with the actual path to your ChromeDriver)
driver = webdriver.Chrome()

# Open the login page
driver.get("http://localhost:8000/accounts/login/")
print("Opened the login page.")

# Maximize the browser window
driver.maximize_window()

# Wait for the page to load completely
time.sleep(3)  # Increased time to ensure the page is fully loaded

# Wait for the username field to be visible and interact with it
wait = WebDriverWait(driver, 10)  # 10-second timeout

try:
    # Step 1: Log in
    username_field = wait.until(EC.visibility_of_element_located((By.NAME, "username")))
    username_field.clear()
    username_field.send_keys('Aneeta')
    print("Entered username.")
    time.sleep(1)  # Time gap between operations

    password_field = wait.until(EC.visibility_of_element_located((By.NAME, 'password')))
    password_field.clear()
    password_field.send_keys('Aneeta@123')
    print("Entered password.")
    time.sleep(1)  # Time gap between operations

    # Wait a moment to ensure fields aren't cleared by JavaScript
    time.sleep(1)

    login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[type="submit"].submit')))
    print("Login button found. Clicking the button.")
    login_button.click()
    time.sleep(3)  # Wait for login to process

    WebDriverWait(driver, 10).until(lambda d: "dashboard" in d.current_url or "success_page" in d.current_url)
    if "dashboard" in driver.current_url or "success_page" in driver.current_url:
        print("Login successful.")
    else:
        raise Exception(f"Login failed. Current URL: {driver.current_url}")

    # Step 2: Navigate to the donate waste page
    driver.get("http://localhost:8000/donors/donate/")
    print("Navigated to the donate waste page.")
    time.sleep(3)  # Wait for the page to load completely

    # Step 3: Fill out the form
    # Locate and click the "Medium of Waste" dropdown to open it
    dropdown = wait.until(EC.element_to_be_clickable((By.NAME, 'medium_of_waste')))
    dropdown.click()
    print("Dropdown clicked.")
    time.sleep(2)  # Time gap to ensure dropdown options are visible

    # Wait for options to be visible and select the desired one
    options = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//select[@name="medium_of_waste"]/option')))
    option_found = False
    for option in options:
        if option.text == 'newspaper':  # Replace with desired option text
            option.click()
            print("Selected newspaper as medium of waste.")
            option_found = True
            break
    if not option_found:
        error_message = f"The selected medium of waste is not available in the dropdown. "
        raise Exception(error_message)
    time.sleep(2)  # Time gap to ensure the selection is processed

    # Locate and fill in the "Quantity" field
    quantity_field = wait.until(EC.visibility_of_element_located((By.NAME, 'quantity')))
    quantity_field.clear()
    quantity = '-7'  # Example quantity
    quantity_field.send_keys(quantity)
    print("Entered quantity.")
    time.sleep(2)  # Time gap to ensure field entry is processed

    # Validate the quantity field value
    if int(quantity) <= 0:
        error_message = f"Quantity must be a positive number. "
        raise ValueError(error_message)
        

    # Locate and fill in the "Location" field
    location_field = wait.until(EC.visibility_of_element_located((By.NAME, 'location')))
    location_field.clear()
    location_field.send_keys('Kochi')  # Example location
    print("Entered location.")
    time.sleep(2)  # Time gap to ensure field entry is processed

    # Locate and upload multiple image files
    image_field = wait.until(EC.visibility_of_element_located((By.ID, 'images')))
    
    # Provide paths to multiple image files
    image_file_paths = [
        r"C:\Users\aleen\OneDrive\Pictures\np1.jpg",
        r"C:\Users\aleen\OneDrive\Pictures\np2.jpg"
    ]

    # Join file paths with newline characters for multiple file upload
    image_file_paths_string = '\n'.join(image_file_paths)
    image_field.send_keys(image_file_paths_string)
    print("Uploaded images.")
    time.sleep(3)  # Wait for images to be processed

    # Step 4: Submit the form
    submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
    submit_button.click()
    print("Form submitted.")
    time.sleep(3)  # Wait for form submission to process

    # Step 5: Verify the form submission success
    WebDriverWait(driver, 10).until(lambda d: "success_page" in d.current_url)
    if "success_page" in driver.current_url:
        print("Donation submission successful.")
    else:
        raise Exception(f"Donation submission failed. Current URL: {driver.current_url}")

except ValueError as ve:
    print(f"Validation error: {ve}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Delay closing of the browser for 10 seconds
    time.sleep(3)
    driver.quit()
