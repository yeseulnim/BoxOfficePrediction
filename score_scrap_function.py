from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

#movie_name = "괴물"
#movie_year = "2006"
#movie_country = "한국"

def get_star_rating(movie_name, movie_year, movie_country):
    URL = f"https://pedia.watcha.com/ko-KR/search?domain=movie&query={movie_name}"

    driver = webdriver.Chrome()
    driver.get(URL)
    star_rating = 0
    try:
        # Wait for the movie list to load
        movie_list = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "koUbE.StyledHorizontalUl.z00m1.iPkT0"))
        )

        # Find all movie title elements
        movie_titles = movie_list.find_elements(By.CLASS_NAME, "fyDfz.qz3Mr")
        movie_info = movie_list.find_elements(By.CLASS_NAME, "zal_L.G41oJ")
        movie_check = movie_list.find_elements(By.CLASS_NAME, "zal_L.ma29X")


        # Extract and print movie names
        if len(movie_titles) == 1:
            movie_titles[0].click()
        else:
            for idx, title in enumerate(movie_titles):
                info = movie_info[idx]
                check = movie_check[idx]
                if title.text == movie_name and info.text.startswith(f"{movie_year}") and check.text == "영화":
                    title.click()
                    break

        star_rating = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "bunOX.QCY16"))
        ).text

    except TimeoutException:
        print("Timed out waiting for page to load")
    except NoSuchElementException:
        print("Could not find the element")
    finally:
        driver.quit()
    return star_rating


#print(get_star_rating("괴불","2006","한국"))





'''
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

movie_name = "괴물"
URL = f"https://pedia.watcha.com/ko-KR/search?domain=movie&query={movie_name}"

driver = webdriver.Chrome()
driver.get(URL)

# Function to find element with explicit wait
def find_element_with_wait(driver, by, value, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except TimeoutException:
        print(f"Element not found: {by}={value}")
        return None

# Find elements
try:
    root = find_element_with_wait(driver, By.ID, "root")
    if root:
        class1 = find_element_with_wait(root, By.CLASS_NAME, "KrZlw")
        if class1:
            class2 = find_element_with_wait(class1, By.CLASS_NAME, "VIvQq.heSEi.MGwoJ")
            if class2:
                class3 = find_element_with_wait(class2, By.CLASS_NAME, "TVbjg")
                if class3:
                    class4 = class3.find_element(by=By.CLASS_NAME, value="WCogg")
                    if class4:
                        class5 = class4.find_element(by=By.CLASS_NAME, value="XaLX9")
                        if class5:
                            class6 = class5.find_element(by=By.CLASS_NAME, value="JeeFJ")
                            if class6:
                                class7 = class6.find_element(by=By.CLASS_NAME, value="Kb9Sb")
                                if class7:
                                    class8 = class7.find_element(by=By.CLASS_NAME, value="wC9Ps")
                                    if class8:
                                        class9 = class8.find_element(by=By.CLASS_NAME, value="risud")
                                        if class9:
                                            class10 = class9.find_element(by=By.CLASS_NAME, value="qXELy")
                                            if class10:
                                                class11 = class10.find_element(by=By.CLASS_NAME,
                                                                               value="koUbE")
                                                if class11:
                                                    class12 = class11.find_elements(by=By.CLASS_NAME, value="_DV31")

                # Continue with the rest of your element finding logic
                # ...

    # If you reach this point, all elements were found successfully
    print("All elements found successfully")

except NoSuchElementException as e:
    print(f"Element not found: {e}")

finally:
    # Optionally, print page source if elements are not found
    print(class12)
    # print(driver.page_source)

    for element in class12:
        try:
            element.find_element(by = By.XPATH, value =  '//*[@title="괴물"]').click()
        except:
            print("Error!")

    driver.quit()

class12
'''

'''
import time
from selenium import webdriver
from selenium.webdriver.common.by import By

URL = f"https://pedia.watcha.com/ko-KR/search?domain=movie&query={"괴물"}"
driver = webdriver.Chrome()
driver.get(URL)
driver.implicitly_wait(5000)

root = driver.find_element(by=By.ID, value = "root")
time.sleep(10)
class1 = root.find_element(by=By.CLASS_NAME, value = "KrZlw")
print(class1)
class2 = class1.find_element(by=By.CLASS_NAME, value = "VIvQq heSEi MGwoJ")
#print(class2)
class3 = class2.find_element(by=By.CLASS_NAME, value = "TVbjg")
#print(class3)
class4 = class3.find_element(by=By.CLASS_NAME, value = "WCogg")
#print(class4)
class5 = class4.find_element(by=By.CLASS_NAME, value = "XaLX9")
class6 = class5.find_element(by=By.CLASS_NAME, value = "JeeFJ")
class7 = class6.find_element(by=By.CLASS_NAME, value = "Kb9Sb")
class8 = class7.find_element(by=By.CLASS_NAME, value = "wC9Ps")
class9 = class8.find_element(by=By.CLASS_NAME, value = "risud")
class10 = class9.find_element(by=By.CLASS_NAME, value = "qXELy")
class11 = class10.find_element(by=By.CLASS_NAME, value = "koUbE StyledHorizontalUl z00m1 iPkT0")
class12 = class11.find_elements(by=By.CLASS_NAME, value = "_DV31")

print(class12)

driver.quit()
'''
