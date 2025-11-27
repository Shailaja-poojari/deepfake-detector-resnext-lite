import json
import hashlib
import os

DB = "users.json"

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def signup(username, password):
    users = json.load(open(DB, "r")) if os.path.exists(DB) else {}

    if username in users:
        return False, "User already exists"

    users[username] = _hash(password)
    json.dump(users, open(DB, "w"))
    return True, "Signup successful!"

def login(username, password):
    if not os.path.exists(DB):
        return False, "No users registered"

    users = json.load(open(DB, "r"))

    if username not in users:
        return False, "Username not found"

    if users[username] != _hash(password):
        return False, "Incorrect password"

    return True, "Login successful!"
