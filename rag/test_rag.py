
import pytest  # -- might need it for fixture, but I assume not

# =================== SETUP =================== #

@pytest.fixture
def thing():
    thing = False
    return thing


# ============================================= #


# =================== TESTS =================== #

def test_hello_world(thing):
    print ("-- other text here")
    assert thing == True


# ============================================= #
