from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


driver = webdriver.Chrome()
driver.get("http://localhost:8000/accounts/login/")
print("Opened the login page.")
driver.maximize_window()
time.sleep(3)  
wait = WebDriverWait(driver, 10)  

try:
   
    username_field = wait.until(EC.visibility_of_element_located((By.NAME, "username")))
    username_field.clear()
    username_field.send_keys('Anju')
    print("Entered username.")
    time.sleep(1) 

    password_field = wait.until(EC.visibility_of_element_located((By.NAME, 'password')))
    password_field.clear()
    password_field.send_keys('Thekkekara@123')
    print("Entered password.")
    time.sleep(1)  
    time.sleep(1)

    login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[type="submit"].submit')))
    print("Login button found. Clicking the button.")
    login_button.click()
    time.sleep(3)

    WebDriverWait(driver, 10).until(lambda d: "dashboard" in d.current_url or "success_page" in d.current_url)
    if "dashboard" in driver.current_url or "success_page" in driver.current_url:
        print("Login successful.")
    else:
        raise Exception(f"Login failed. Current URL: {driver.current_url}")

    
    driver.get("http://localhost:8000/donors/donate/")
    print("Navigated to the donate waste page.")
    time.sleep(3) 
    dropdown = wait.until(EC.element_to_be_clickable((By.NAME, 'medium_of_waste')))
    dropdown.click()
    print("Dropdown clicked.")
    time.sleep(2) 

    options = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//select[@name="medium_of_waste"]/option')))
    option_found = False
    for option in options:
        if option.text == 'Cloth':  
            option.click()
            print("Selected newspaper as medium of waste.")
            option_found = True
            break
    if not option_found:
        error_message = f"The selected medium of waste is not available in the dropdown. "
        raise Exception(error_message)
    time.sleep(2)  

    quantity_field = wait.until(EC.visibility_of_element_located((By.NAME, 'quantity')))
    quantity_field.clear()
    quantity = '15'  
    quantity_field.send_keys(quantity)
    print("Entered quantity.")
    time.sleep(2) 

    
    if int(quantity) <= 0:
        error_message = f"Quantity must be a positive number. "
        raise ValueError(error_message)
        

    
    location_field = wait.until(EC.visibility_of_element_located((By.NAME, 'location')))
    location_field.clear()
    location_field.send_keys('Ernakulam ') 
    print("Entered location.")
    time.sleep(2)  

   
    image_field = wait.until(EC.visibility_of_element_located((By.ID, 'images')))
    

    image_file_paths = [
        r"C:\Users\aleen\OneDrive\Pictures\cloth.jpg",
        r"C:\Users\aleen\OneDrive\Pictures\cloth1.jpg"
    ]

    
    image_file_paths_string = '\n'.join(image_file_paths)
    image_field.send_keys(image_file_paths_string)
    print("Uploaded images.")
    time.sleep(3) 

    
    submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
    submit_button.click()
    print("Form submitted.")
    time.sleep(3)  

    
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
    
    time.sleep(3)
    driver.quit()
