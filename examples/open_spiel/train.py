from dotenv import load_dotenv
from moozi.driver import Driver, get_config

load_dotenv()
config = get_config()
driver = Driver.setup(config)
driver.start()
driver.run()
