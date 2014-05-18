# -*- coding: utf-8 -*-

import pytest

from .system import PolynomialSystem, FourierSystem, ConstantSystem

@pytest.fixture(params = [
        PolynomialSystem(
                dz = 1e-2,
                dt = 1.0,
                a = 1.0,
                nx = 3,
                nz = 4,
                Ra = 1.0,
                Pr = 1.0,
            ),
        FourierSystem(
                dz = 1e-2,
                dt = 1.0,
                a = 1.0,
                nx = 3,
                nz = 4,
                Ra = 1.0,
                Pr = 1.0,
            ),
        PolynomialSystem(
                dz = 1e-2,
                dt = 2.0,
                a = 0.5,
                nx = 10,
                nz = 8,
                Ra = 5.0,
                Pr = 10.0,
            ),
        FourierSystem(
                dz = 1e-2,
                dt = 2.0,
                a = 0.5,
                nx = 10,
                nz = 8,
                Ra = 5.0,
                Pr = 10.0,
            ),
        ConstantSystem(
            dz = 1e-2,
            dt = 1.0,
            a = 1.0,
            nx = 3,
            nz = 4,
            Ra = 1.0,
            Pr = 1.0,
        ),
        ConstantSystem(
            dz = 1e-2,
            dt = 2.0,
            a = 0.5,
            nx = 10,
            nz = 8,
            Ra = 5.0,
            Pr = 10.0,
        )
    ])
def system(request):
    """Generate individual systems."""
    return request.param
    
@pytest.fixture(params=["Temperature", "Vorticity", "Stream"])
def fluid_componet(request):
    """Parameterized across the fluid components."""
    return request.param