import math

def test_squrt():
    num = 81
    assert math.sqrt(num)==9

def test_cube():
    num = 3
    assert num**3 == 27

def test_sum():
    num = 2
    assert num + num != 5
