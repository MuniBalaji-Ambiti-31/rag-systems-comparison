# -*- coding: utf-8 -*-
# RAGMathFunctions.py — simple math tools used by the ReAct agent

from __future__ import annotations
import math

def multiply(a: float, b: float) -> float:
    return float(a) * float(b)

def add(a: float, b: float) -> float:
    return float(a) + float(b)

def subtract(a: float, b: float) -> float:
    return float(a) - float(b)

def divide(a: float, b: float) -> float:
    b = float(b)
    if b == 0.0:
        raise ZeroDivisionError("Division by zero")
    return float(a) / b

def compute_circle_area(r: float) -> float:
    r = float(r)
    if r < 0:
        raise ValueError("Radius must be non-negative")
    return math.pi * r * r