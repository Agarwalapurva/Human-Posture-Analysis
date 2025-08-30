Human-Posture-Analysis

Overview

A posture monitoring system using computer vision to assess ergonomics. It calculates RULA (upper body) and REBA (whole body) scores from joint angles and visualizes posture progress on an interactive dashboard.

Features:
-Real-time posture detection via webcam
-Joint angle calculations using Mediapipe
-Ergonomic scoring (RULA & REBA)
-Dashboard with score trends, session filters, and improvement summary

Tech Stack:
Python, OpenCV, Mediapipe, NumPy, Pandas, Streamlit

Steps to install:
Clone and install: 
-git clone https://github.com/yourusername/posture-analysis.git
-cd posture-analysis
-pip install -r requirements.txt

Run detection: python main.py

Run Dashboard: streamlit run dashboard.py
