from dotenv import load_dotenv
from moozi.driver import Driver, get_config, ConfigFactory

load_dotenv()
config = get_config()
config_factory = ConfigFactory(config)
driver = Driver.setup(config)
driver.start()
driver.run()
