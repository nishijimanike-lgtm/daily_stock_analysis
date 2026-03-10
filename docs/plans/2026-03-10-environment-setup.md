# Environment Setup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up a local development environment for the `daily_stock_analysis` project.

**Architecture:** Create a project-specific Python virtual environment, install dependencies, and configure the `.env` file using the project's default ports.

**Tech Stack:** Python 3.10+, Node.js/npm, Vite, FastAPI.

---

### Task 1: Initialize Python Virtual Environment

**Step 1: Create virtual environment**
Run: `python -m venv .venv`

**Step 2: Upgrade pip**
Run: `.\.venv\Scripts\python.exe -m pip install --upgrade pip`

**Step 3: Install requirements**
Run: `.\.venv\Scripts\pip.exe install -r requirements.txt`

### Task 2: Configure Environment Variables

**Step 1: Create .env from template**
Run: `copy .env.example .env`

**Step 2: Verify .env exists**
Run: `ls .env`

### Task 3: Setup Frontend

**Step 1: Install frontend dependencies**
Run: `cd apps/dsa-web && npm install`

### Task 4: Verification

**Step 1: Verify Python environment**
Run: `.\.venv\Scripts\python.exe test_env.py`

**Step 2: Start Web UI (Dry Run)**
Run: `.\.venv\Scripts\python.exe main.py --webui-only`
