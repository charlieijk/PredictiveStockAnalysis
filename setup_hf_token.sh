#!/bin/bash

# Setup HuggingFace Token
# This script helps you configure your HF token

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   HuggingFace Token Setup                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "To get your token:"
echo "1. Go to: https://huggingface.co/settings/tokens"
echo "2. Create a new token or copy existing one"
echo "3. Make sure it has WRITE permissions"
echo ""
echo "Enter your HuggingFace token (starts with 'hf_'):"
read -r HF_TOKEN

# Validate token format
if [[ ! $HF_TOKEN == hf_* ]]; then
    echo "⚠️  Warning: Token doesn't start with 'hf_'. Are you sure it's correct?"
fi

# Detect shell
SHELL_CONFIG=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    SHELL_CONFIG="$HOME/.profile"
fi

echo ""
echo "Options:"
echo "1) Set for this session only (temporary)"
echo "2) Save permanently to $SHELL_CONFIG (recommended)"
echo ""
echo "Enter choice [1 or 2]:"
read -r CHOICE

if [[ $CHOICE == "1" ]]; then
    # Temporary
    export HF_TOKEN="$HF_TOKEN"
    echo ""
    echo "✓ Token set for this session!"
    echo "You can now run:"
    echo "  python push_to_hf_git.py charlieijk/SolarRegatta"
    echo ""
    echo "Note: This will be lost when you close the terminal."

elif [[ $CHOICE == "2" ]]; then
    # Permanent

    # Check if already in config
    if grep -q "export HF_TOKEN=" "$SHELL_CONFIG" 2>/dev/null; then
        echo ""
        echo "HF_TOKEN already exists in $SHELL_CONFIG"
        echo "Update it? [y/n]:"
        read -r UPDATE

        if [[ $UPDATE == "y" ]]; then
            # Remove old line and add new one
            sed -i.bak '/export HF_TOKEN=/d' "$SHELL_CONFIG"
            echo "export HF_TOKEN=\"$HF_TOKEN\"" >> "$SHELL_CONFIG"
            echo "✓ Updated HF_TOKEN in $SHELL_CONFIG"
        fi
    else
        # Add to config
        echo "" >> "$SHELL_CONFIG"
        echo "# HuggingFace Token" >> "$SHELL_CONFIG"
        echo "export HF_TOKEN=\"$HF_TOKEN\"" >> "$SHELL_CONFIG"
        echo "✓ Added HF_TOKEN to $SHELL_CONFIG"
    fi

    # Also set for current session
    export HF_TOKEN="$HF_TOKEN"

    echo ""
    echo "✓ Token saved permanently!"
    echo ""
    echo "To activate now, run:"
    echo "  source $SHELL_CONFIG"
    echo ""
    echo "Or open a new terminal window."

else
    echo "Invalid choice. Exiting."
    exit 1
fi

echo ""
echo "Test your token:"
echo "  python -c 'import os; print(\"Token:\", os.getenv(\"HF_TOKEN\")[:10] + \"...\")'"
