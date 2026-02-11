# Password Authentication Setup Guide

The SEC Financial Data Extractor includes optional password protection. When enabled, users must enter a password before accessing the application.

## 🔒 How It Works

- **Optional:** Authentication is disabled by default
- **Simple:** Single password for all users
- **Secure:** Password stored in secrets/environment variables (never in code)
- **Session-based:** Once authenticated, stays logged in for the session

---

## 📝 Setup Instructions

### Option 1: Local Development

Set the `APP_PASSWORD` environment variable before running the app:

**Windows (Command Prompt):**
```cmd
set APP_PASSWORD=YourSecurePassword123
streamlit run app.py
```

**Windows (PowerShell):**
```powershell
$env:APP_PASSWORD="YourSecurePassword123"
streamlit run app.py
```

**Mac/Linux:**
```bash
export APP_PASSWORD="YourSecurePassword123"
streamlit run app.py
```

### Option 2: Streamlit Cloud Deployment

1. Deploy your app to Streamlit Cloud
2. Go to your app's settings
3. Navigate to the **Secrets** section
4. Add the following:

```toml
APP_PASSWORD = "YourSecurePassword123"
OPENAI_API_KEY = "sk-your-openai-key-here"
```

5. Save and the app will restart with authentication enabled

### Option 3: Using .streamlit/secrets.toml (Local)

For local development with persistent secrets:

1. Create a file `.streamlit/secrets.toml` in the project directory
2. Add your password:

```toml
APP_PASSWORD = "YourSecurePassword123"
OPENAI_API_KEY = "sk-your-openai-key-here"
```

3. Run the app normally: `streamlit run app.py`

**Note:** `secrets.toml` is already in `.gitignore` and won't be committed to git.

### Option 4: Docker Deployment

Pass the password as an environment variable:

```bash
docker run -p 8501:8501 \
  -e APP_PASSWORD="YourSecurePassword123" \
  -e OPENAI_API_KEY="sk-your-key" \
  sec-extractor
```

---

## 🔓 Disabling Authentication

To run the app **without** password protection:

- Simply **don't set** the `APP_PASSWORD` variable
- The app will automatically allow access without authentication

---

## 🔐 Security Best Practices

### ✅ DO:
- Use strong, unique passwords (12+ characters)
- Store password in secrets/environment variables only
- Change password regularly
- Use different passwords for dev/staging/production

### ❌ DON'T:
- Never hardcode passwords in the code
- Never commit secrets.toml to git
- Don't share passwords in chat/email (use secure channels)
- Don't use simple passwords like "password123"

---

## 🎯 Example Strong Passwords

Good examples:
- `SecFinance2026!Data`
- `Extract#Financial$2026`
- `Capital!Markets#Analysis9`

---

## 🧪 Testing Authentication

1. **With password set:**
   ```bash
   export APP_PASSWORD="TestPassword123"
   streamlit run app.py
   ```
   - You should see a login screen
   - Enter "TestPassword123" to access

2. **Without password:**
   ```bash
   # Don't set APP_PASSWORD
   streamlit run app.py
   ```
   - App opens directly without login

---

## 🔄 Changing the Password

### Local Development:
1. Close the app
2. Change the environment variable or `secrets.toml`
3. Restart the app
4. Users will need to log in again with new password

### Streamlit Cloud:
1. Go to app settings → Secrets
2. Update the `APP_PASSWORD` value
3. Save (app will auto-restart)
4. All users will be logged out and need new password

---

## 🛠️ Troubleshooting

### "Incorrect password" but password is correct
- Check for extra spaces or hidden characters
- Environment variable might not be set correctly
- Try restarting the app

### App asks for password but you didn't set one
- Check if APP_PASSWORD is set in environment
- Check `.streamlit/secrets.toml` file
- Unset the variable if you don't want authentication

### Session expires too quickly
- This is controlled by Streamlit's session management
- Users need to re-authenticate if they close the browser
- Sessions persist during page refreshes

---

## 📋 Quick Reference

| Deployment | How to Set Password |
|-----------|-------------------|
| **Local Dev** | `export APP_PASSWORD="password"` |
| **Streamlit Cloud** | Add to Secrets in web dashboard |
| **Docker** | `-e APP_PASSWORD="password"` |
| **Heroku** | `heroku config:set APP_PASSWORD=password` |
| **Disable Auth** | Don't set APP_PASSWORD at all |

---

## 🔍 How to Check If Authentication is Active

When you start the app:

**Authentication ENABLED:**
- Login screen appears immediately
- Shows "🔒 SEC Financial Data Extractor"
- Password input field visible

**Authentication DISABLED:**
- Main app interface appears
- No login screen
- Direct access to features

---

## 💡 Tips

1. **For demos/testing:** Use a simple password like "demo123"
2. **For production:** Use a strong, unique password
3. **For development:** Consider disabling authentication
4. **For shared environments:** Enable authentication with a secure password

---

**Need Help?**
Refer to DEPLOYMENT.md for general deployment instructions.
