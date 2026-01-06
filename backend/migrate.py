#!/usr/bin/env python3
"""
Database migration management script for DataPrep AI Platform.
"""
import os
import sys
import subprocess
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.database_utils import check_database_connection, initialize_database


def run_alembic_command(command: str) -> bool:
    """Run an Alembic command."""
    try:
        result = subprocess.run(
            f"alembic {command}",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running alembic {command}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    """Main migration management function."""
    if len(sys.argv) < 2:
        print("Usage: python migrate.py [command]")
        print("Commands:")
        print("  check     - Check database connection")
        print("  init      - Initialize database with tables")
        print("  upgrade   - Run Alembic upgrade to head")
        print("  downgrade - Run Alembic downgrade")
        print("  revision  - Create new Alembic revision")
        print("  current   - Show current revision")
        print("  history   - Show migration history")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "check":
        success = check_database_connection()
        if success:
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
        sys.exit(0 if success else 1)
    
    elif command == "init":
        print("Initializing database...")
        success = initialize_database()
        if success:
            print("✅ Database initialized successfully")
        else:
            print("❌ Database initialization failed")
        sys.exit(0 if success else 1)
    
    elif command == "upgrade":
        print("Running Alembic upgrade...")
        success = run_alembic_command("upgrade head")
        if success:
            print("✅ Database upgrade completed")
        else:
            print("❌ Database upgrade failed")
        sys.exit(0 if success else 1)
    
    elif command == "downgrade":
        if len(sys.argv) < 3:
            revision = "-1"  # Downgrade one step
        else:
            revision = sys.argv[2]
        
        print(f"Running Alembic downgrade to {revision}...")
        success = run_alembic_command(f"downgrade {revision}")
        if success:
            print("✅ Database downgrade completed")
        else:
            print("❌ Database downgrade failed")
        sys.exit(0 if success else 1)
    
    elif command == "revision":
        if len(sys.argv) < 3:
            print("Usage: python migrate.py revision 'message'")
            sys.exit(1)
        
        message = sys.argv[2]
        print(f"Creating new revision: {message}")
        success = run_alembic_command(f'revision --autogenerate -m "{message}"')
        if success:
            print("✅ New revision created")
        else:
            print("❌ Failed to create revision")
        sys.exit(0 if success else 1)
    
    elif command == "current":
        print("Current database revision:")
        run_alembic_command("current")
    
    elif command == "history":
        print("Migration history:")
        run_alembic_command("history")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()