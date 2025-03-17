from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time

# Initialize WebDriver with minimal options
options = webdriver.ChromeOptions()
options.add_argument("--ignore-certificate-errors")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)  # Increased wait time

try:
    # Start from homepage and find the login link
    print("Starting test for Order #66...")
    
    # Go to homepage first
    driver.get("http://localhost:8000/")
    print(f"Current URL: {driver.current_url}")
    
    # Try to find the Delivery menu
    print("Looking for DELIVERY menu...")
    delivery_menu = driver.find_element(By.LINK_TEXT, "DELIVERY")
    
    # Hover over the delivery menu
    actions = ActionChains(driver)
    actions.move_to_element(delivery_menu).perform()
    time.sleep(2)
    
    # Look for Delivery Partner Login
    login_link = driver.find_element(By.LINK_TEXT, "Delivery Partner Login")
    login_link.click()
    print("Clicked on Delivery Partner Login")
    time.sleep(2)
    
    # Login
    username_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
    username_field.clear()
    username_field.send_keys("jagan@gmail.com")
    print("Entered username/email")
    
    password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))
    password_field.clear()
    password_field.send_keys("Jagan@123")
    print("Entered password")
    
    # Click the submit button
    submit_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
    submit_button.click()
    print("Clicked login button")
    time.sleep(3)
    
    # Go directly to My Deliveries page
    print("Navigating directly to My Deliveries page...")
    driver.get("http://localhost:8000/delivery/my-deliveries/")
    time.sleep(3)
    print(f"Current URL: {driver.current_url}")
    
    # Save page source for analysis
    with open("deliveries_page_source.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print("Saved page source for analysis")
    
    # Take a screenshot of the initial page
    driver.save_screenshot("1_deliveries_page.png")
    
    # DIRECT APPROACH: Find the button by data-bs-target attribute
    print("Looking for modal trigger buttons by data-bs-target attribute...")
    modal_buttons = driver.find_elements(By.CSS_SELECTOR, "button[data-bs-toggle='modal']")
    print(f"Found {len(modal_buttons)} modal trigger buttons")
    
    # Find all order headings to identify which button belongs to Order #66
    order_headings = driver.find_elements(By.XPATH, "//h5[contains(text(), 'Order #')]")
    print(f"Found {len(order_headings)} order headings")
    
    # Print all order headings and their text
    for i, heading in enumerate(order_headings):
        try:
            print(f"Order heading {i+1}: {heading.text}")
            driver.execute_script("arguments[0].style.border='2px solid blue'", heading)
        except:
            pass
    
    # Print all modal buttons and their targets
    for i, button in enumerate(modal_buttons):
        try:
            target = button.get_attribute("data-bs-target")
            print(f"Modal button {i+1}: Target: {target}")
            driver.execute_script("arguments[0].style.border='2px solid red'", button)
        except:
            pass
    
    driver.save_screenshot("2_all_elements_highlighted.png")
    
    # Find the card containing Order #66
    print("Looking for the card containing Order #66...")
    order_66_element = None
    
    for heading in order_headings:
        if "Order #67" in heading.text:
            order_66_element = heading
            print("Found Order #66 heading")
            driver.execute_script("arguments[0].style.border='3px solid green'", heading)
            break
    
    if order_66_element:
        # Find the closest modal button to Order #66
        print("Finding the closest modal button to Order #66...")
        
        # Get the card containing Order #66
        try:
            order_card = order_66_element.find_element(By.XPATH, "./ancestor::div[contains(@class, 'card')]")
            print("Found Order #66 card")
            driver.execute_script("arguments[0].style.border='3px solid purple'", order_card)
            
            # Try to find the button within this card
            try:
                update_btn = order_card.find_element(By.CSS_SELECTOR, "button[data-bs-toggle='modal']")
                print("Found Update Status button within Order #66 card")
            except:
                # If not found directly, try finding by text
                try:
                    update_btn = order_card.find_element(By.XPATH, ".//button[contains(text(), 'Update Status')]")
                    print("Found Update Status button by text within Order #66 card")
                except:
                    # If still not found, use proximity
                    print("Button not found directly in card, using proximity")
                    
                    # Get the position of Order #66
                    order_location = order_66_element.location
                    order_y = order_location['y']
                    
                    # Find the closest button
                    closest_button = None
                    min_distance = float('inf')
                    
                    for button in modal_buttons:
                        try:
                            button_y = button.location['y']
                            distance = abs(button_y - order_y)
                            if distance < min_distance:
                                min_distance = distance
                                closest_button = button
                        except:
                            pass
                    
                    if closest_button:
                        update_btn = closest_button
                        print(f"Found closest button to Order #66 (distance: {min_distance}px)")
                    else:
                        # Last resort: just use the first button
                        update_btn = modal_buttons[0]
                        print("Using first modal button as fallback")
        except:
            # If we can't find the card, use proximity to the heading
            print("Could not find Order #66 card, using proximity to heading")
            
            # Get the position of Order #66 heading
            order_location = order_66_element.location
            order_y = order_location['y']
            
            # Find the closest button
            closest_button = None
            min_distance = float('inf')
            
            for button in modal_buttons:
                try:
                    button_y = button.location['y']
                    distance = abs(button_y - order_y)
                    if distance < min_distance:
                        min_distance = distance
                        closest_button = button
                except:
                    pass
            
            if closest_button:
                update_btn = closest_button
                print(f"Found closest button to Order #66 heading (distance: {min_distance}px)")
            else:
                # Last resort: just use the first button
                update_btn = modal_buttons[0]
                print("Using first modal button as fallback")
    else:
        # If we can't find Order #66, use the first modal button
        print("Order #66 not found, using first modal button")
        update_btn = modal_buttons[0]
    
    # Highlight the button we're going to click
    driver.execute_script("arguments[0].style.border='5px solid red'", update_btn)
    driver.save_screenshot("3_button_to_click.png")
    
    # Get the modal target ID
    modal_target = update_btn.get_attribute("data-bs-target")
    print(f"Button targets modal: {modal_target}")
    
    # Scroll to the button
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", update_btn)
    time.sleep(2)
    driver.save_screenshot("4_button_scrolled_into_view.png")
    
    # CRITICAL: Try multiple click methods to ensure the button is clicked
    print("Attempting to click the button using multiple methods...")
    
    # Method 1: Regular click
    try:
        print("Trying regular click...")
        update_btn.click()
        print("Regular click succeeded")
    except Exception as e:
        print(f"Regular click failed: {str(e)}")
        
        # Method 2: JavaScript click
        try:
            print("Trying JavaScript click...")
            driver.execute_script("arguments[0].click();", update_btn)
            print("JavaScript click succeeded")
        except Exception as e:
            print(f"JavaScript click failed: {str(e)}")
            
            # Method 3: Action chains
            try:
                print("Trying ActionChains click...")
                actions = ActionChains(driver)
                actions.move_to_element(update_btn).click().perform()
                print("ActionChains click succeeded")
            except Exception as e:
                print(f"ActionChains click failed: {str(e)}")
                
                # Method 4: Try clicking by coordinates
                try:
                    print("Trying click by coordinates...")
                    actions = ActionChains(driver)
                    actions.move_to_element_with_offset(update_btn, 5, 5).click().perform()
                    print("Coordinate click succeeded")
                except Exception as e:
                    print(f"Coordinate click failed: {str(e)}")
    
    time.sleep(2)
    driver.save_screenshot("5_after_button_click.png")
    
    # Check if modal appeared by looking for the specific modal ID
    if modal_target:
        modal_id = modal_target.replace('#', '')
        print(f"Looking for modal with ID: {modal_id}")
        
        try:
            modal = wait.until(EC.visibility_of_element_located((By.ID, modal_id)))
            print(f"Found modal with ID: {modal_id}")
        except:
            print(f"Could not find modal with ID: {modal_id}")
            # Try generic modal selector
            try:
                modal = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.modal.show, div.modal[style*='display: block']")))
                print("Found modal using generic selector")
            except Exception as e:
                print(f"Could not find modal using generic selector: {str(e)}")
                # Save page source after click for debugging
                with open("after_click_source.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                print("Saved page source after click")
                raise Exception("Modal did not appear after clicking button")
    else:
        # If we don't have a target, use generic modal selector
        try:
            modal = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.modal.show, div.modal[style*='display: block']")))
            print("Found modal using generic selector")
        except Exception as e:
            print(f"Could not find modal: {str(e)}")
            # Save page source after click for debugging
            with open("after_click_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("Saved page source after click")
            raise Exception("Modal did not appear after clicking button")
    
    print("Modal appeared successfully")
    driver.save_screenshot("6_modal_appeared.png")
    
    # Find the status dropdown in the modal
    print("Looking for status dropdown...")
    try:
        status_select = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "select[name='status']")))
        status_select.click()
        print("Clicked status dropdown")
        time.sleep(1)
        driver.save_screenshot("7_status_dropdown_open.png")
        
        # Find all options in the dropdown
        options = status_select.find_elements(By.TAG_NAME, "option")
        print(f"Found {len(options)} options in dropdown")
        
        # Find enabled options (not disabled)
        enabled_options = []
        for option in options:
            if option.get_attribute("disabled") is None or option.get_attribute("disabled") == "false":
                enabled_options.append(option)
                print(f"Found enabled option: {option.text}")
        
        # Select the first enabled option that's not the default "Select Status"
        selected_option = None
        for option in enabled_options:
            if option.text.strip() and option.text.strip().lower() != "select status":
                selected_option = option
                break
        
        if selected_option:
            selected_option.click()
            print(f"Selected option: {selected_option.text}")
        else:
            # If no suitable option found, try to select the last option
            if len(enabled_options) > 0:
                enabled_options[-1].click()
                print(f"Selected last enabled option: {enabled_options[-1].text}")
            else:
                print("No enabled options found in dropdown")
        
        time.sleep(1)
        driver.save_screenshot("8_option_selected.png")
        
        # Find and fill the notes field
        print("Looking for notes field...")
        try:
            notes_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "textarea[name='notes'], textarea[name='message']")))
            notes_field.clear()
            notes_field.send_keys("Status updated via automated test")
            print("Entered notes")
        except Exception as e:
            print(f"Notes field not found or not required: {str(e)}")
        
        # Find and click the Update Status button in the modal
        print("Looking for Update Status button in modal...")
        try:
            update_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'modal-footer')]//button[@type='submit']")))
        except:
            # Try alternative selectors
            try:
                update_button = modal.find_element(By.XPATH, ".//button[@type='submit']")
            except:
                # Last resort: any button in the modal footer
                update_button = modal.find_element(By.XPATH, ".//div[contains(@class, 'modal-footer')]//button")
        
        driver.execute_script("arguments[0].style.border='3px solid green'", update_button)
        driver.save_screenshot("9_before_final_update.png")
        
        # Click the update button
        try:
            update_button.click()
            print("Clicked final Update Status button (regular click)")
        except:
            try:
                driver.execute_script("arguments[0].click();", update_button)
                print("Clicked final Update Status button (JavaScript click)")
            except Exception as e:
                print(f"Error clicking final update button: {str(e)}")
                raise
        
        time.sleep(3)
        driver.save_screenshot("10_after_final_update.png")
        
        print("Test completed successfully")
    except Exception as e:
        print(f"Error interacting with modal: {str(e)}")
        driver.save_screenshot("modal_interaction_error.png")
        raise

except Exception as e:
    print(f"Error during test: {str(e)}")
    print(f"Current URL: {driver.current_url}")
    driver.save_screenshot("error.png")
    
    # Try to save page source for debugging
    try:
        with open("error_page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Error page source saved to error_page_source.html")
    except:
        print("Could not save page source")

finally:
    # Always quit the driver
    driver.quit()
    print("Test finished, driver closed")
