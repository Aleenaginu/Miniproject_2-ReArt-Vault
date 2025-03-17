from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

# Initialize WebDriver with SSL certificate ignore
options = webdriver.ChromeOptions()
options.add_argument("--ignore-certificate-errors")  # Ignore SSL issues
options.add_argument("--disable-extensions")  # Disable browser extensions
options.add_argument("--disable-popup-blocking")  # Prevent popups from blocking actions
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)

try:
    driver.get("http://localhost:8000/")
    print("Opened the homepage.")
    driver.maximize_window()
    time.sleep(2)

    # Open Shop Page
    driver.get("http://localhost:8000/shop/")
    print("Opened the shop page.")
    time.sleep(2)

    # Navigate to Login Page
    driver.get("http://localhost:8000/shop/customerlogin")
    print("Opened the login page.")
    time.sleep(2)

    # Enter Username
    username_field = wait.until(EC.visibility_of_element_located((By.NAME, "username")))
    username_field.clear()
    username_field.send_keys("Adarsh@2025")
    print("Entered username.")

    # Enter Password
    password_field = wait.until(EC.visibility_of_element_located((By.NAME, "password")))
    password_field.clear()
    password_field.send_keys("Adarsh@123")
    print("Entered password.")

    # Click Login Button
    login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
    login_button.click()
    print("Login button clicked.")

    # Wait for Dashboard
    wait.until(lambda d: "dashboard" in d.current_url or "shop" in d.current_url)
    print("Login successful.")

    # Navigate back to Shop Page
    driver.get("http://localhost:8000/shop/")
    print("Navigated to the shop page.")
    time.sleep(3)  # Give it extra time to load
    
    # Take a screenshot to see the page state
    driver.save_screenshot("shop_page.png")
    print("Saved screenshot of shop page.")

    # Find the profile details container based on the HTML structure
    profile_details = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".profile-details")))
    print("Found profile details container.")
    
    # Take a screenshot before clicking
    driver.execute_script("arguments[0].style.border='3px solid red'", profile_details)
    driver.save_screenshot("profile_details_highlighted.png")
    
    # Click on the profile details to show the dropdown
    actions = ActionChains(driver)
    actions.move_to_element(profile_details).click().perform()
    print("Clicked on profile details to show dropdown.")
    time.sleep(1)  # Wait for dropdown to appear
    
    # Take a screenshot after clicking to see if dropdown appears
    driver.save_screenshot("after_profile_click.png")
    
    # Make dropdown visible using JavaScript (in case hover doesn't work)
    driver.execute_script("arguments[0].querySelector('.dropdown-menu').style.display = 'block';", profile_details)
    print("Made dropdown menu visible using JavaScript.")
    time.sleep(1)
    
    # Take a screenshot to verify dropdown is visible
    driver.save_screenshot("dropdown_visible.png")
    
    # Find and click on "Add Address" in the dropdown
    add_address_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'dropdown-item') and contains(text(), 'Add Address')]")))
    print("Found Add Address link in dropdown.")
    
    # Highlight the Add Address link
    driver.execute_script("arguments[0].style.border='3px solid green'", add_address_link)
    driver.save_screenshot("add_address_highlighted.png")
    
    # Click the Add Address link
    add_address_link.click()
    print("Clicked on Add Address link.")
    
    # Wait for Add Address page to load
    wait.until(lambda d: "add-address" in d.current_url)
    print("Successfully reached Add Address page.")
    
    # Take a screenshot of the Add Address page
    driver.save_screenshot("add_address_page.png")
    
    # Now click on the "ADD A NEW ADDRESS" button
    add_new_address_button = wait.until(EC.element_to_be_clickable((By.ID, "showAddressForm")))
    print("Found 'ADD A NEW ADDRESS' button.")
    
    # Highlight the button
    driver.execute_script("arguments[0].style.border='3px solid blue'", add_new_address_button)
    driver.save_screenshot("add_new_address_button_highlighted.png")
    
    # Click the button
    add_new_address_button.click()
    print("Clicked on 'ADD A NEW ADDRESS' button.")
    time.sleep(1)  # Wait for form to appear
    
    # Take a screenshot to verify form is visible
    driver.save_screenshot("address_form_visible.png")
    
    # Fill out the address form
    # Address Type
    address_type_field = wait.until(EC.visibility_of_element_located((By.ID, "address_type")))
    address_type_field.clear()
    address_type_field.send_keys("Home")
    print("Entered Address Type.")
    
    # Full Name
    full_name_field = wait.until(EC.visibility_of_element_located((By.ID, "full_name")))
    full_name_field.clear()
    full_name_field.send_keys("John Doe")
    print("Entered Full Name.")
    
    # Phone Number
    phone_field = wait.until(EC.visibility_of_element_located((By.ID, "phone")))
    phone_field.clear()
    phone_field.send_keys("1234567890")
    print("Entered Phone Number.")
    
    # Address
    address_field = wait.until(EC.visibility_of_element_located((By.ID, "address")))
    address_field.clear()
    address_field.send_keys("123 Main Street")
    print("Entered Address.")
    
    # City
    city_field = wait.until(EC.visibility_of_element_located((By.ID, "city")))
    city_field.clear()
    city_field.send_keys("New York")
    print("Entered City.")
    
    # State
    state_field = wait.until(EC.visibility_of_element_located((By.ID, "state")))
    state_field.clear()
    state_field.send_keys("NY")
    print("Entered State.")
    
    # ZIP Code
    zip_code_field = wait.until(EC.visibility_of_element_located((By.ID, "zip_code")))
    zip_code_field.clear()
    zip_code_field.send_keys("10001")
    print("Entered ZIP Code.")
    
    # Take a screenshot of the filled form
    driver.save_screenshot("filled_address_form.png")
    
    # Submit the form
    submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.site-btn")))
    print("Found Submit button.")
    
    # Highlight the submit button
    driver.execute_script("arguments[0].style.border='3px solid green'", submit_button)
    driver.save_screenshot("submit_button_highlighted.png")
    
    # Click the submit button
    submit_button.click()
    print("Clicked Submit button.")
    time.sleep(3)  # Wait for form submission to complete
    
    # Take a final screenshot after submission
    driver.save_screenshot("after_form_submission.png")
    
    print("Address form submitted successfully.")

except Exception as e:
    print("Test failed:", str(e))
    driver.save_screenshot("error.png")  # Capture a screenshot on failure
    
    # Print current URL for debugging
    try:
        print("Current URL:", driver.current_url)
    except:
        pass
    
    # Try to save page source for debugging
    try:
        with open("error_page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Error page source saved to error_page_source.html")
    except:
        print("Could not save page source")

finally:
    driver.quit()
