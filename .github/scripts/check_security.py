#!/usr/bin/env python
"""Security and PII checker for pywatershed repository.

This script checks for:
1. Absolute file system paths (e.g., /Users/..., /home/..., C:\\...)
2. IP addresses
3. Internal server hostnames
4. Usernames/passwords or credentials

Can be run in two modes:
- Pre-commit mode: Check only staged files (for pre-commit hook)
- Audit mode: Check all version-controlled files (for full repository audit)

Usage:
    # Pre-commit mode (default - checks staged files)
    python check_security.py

    # Audit mode (checks all tracked files)
    python check_security.py --audit

    # Only check specific types
    python check_security.py --check-paths --check-ips

    # Generate report file
    python check_security.py --audit --output security_report.txt

Customization:
    To modify the checks, edit the SecurityChecker.__init__ method:
    - Add exclusion patterns: Modify `exclude_patterns` list
    - Add allowed paths: Modify `allowed_paths` set
    - Add allowed domains: Modify `allowed_domains` set
    - Modify regex patterns: Edit the pattern strings in each check method
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class SecurityChecker:
    """Check files for security and PII issues."""

    def __init__(
        self,
        check_paths: bool = True,
        check_ips: bool = True,
        check_hostnames: bool = True,
        check_credentials: bool = True,
        exclude_patterns: List[str] = None,
    ):
        """Initialize security checker.

        Args:
            check_paths: Check for absolute file paths
            check_ips: Check for IP addresses
            check_hostnames: Check for internal hostnames
            check_credentials: Check for passwords/credentials
            exclude_patterns: Glob patterns to exclude from checking
        """
        self.do_check_paths = check_paths
        self.do_check_ips = check_ips
        self.do_check_hostnames = check_hostnames
        self.do_check_credentials = check_credentials

        # Default exclusions
        self.exclude_patterns = exclude_patterns or [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.pdf",
            "*.zip",
            "*.tar.gz",
            "*.nc",  # NetCDF files
            "*.pyc",
            "*.so",
            "*.dll",
            "*.dylib",
            "*.lock",
            "*check_security.py",  # The security checker itself
            "*README_security.md",  # Security checker documentation
        ]

        # Allowlisted paths (known safe absolute paths)
        self.allowed_paths = {
            "/usr/bin",
            "/usr/local",
            "/bin",
            "/etc",
            "/opt",
            "/var",
            "/tmp",
            "/dev/null",
        }

        # Allowlisted domains (public, not internal servers)
        self.allowed_domains = {
            "github.com",
            "usgs.gov",  # Public USGS sites
            "doi.gov",
            "doi.org",  # Digital Object Identifier system
            "doi-usgs.github.io",  # pywatershed GitHub Pages documentation
            "sciencebase.gov",
            "waterdata.usgs.gov",
            "waterservices.usgs.gov",
            "pypi.org",
            "conda.io",
            "anaconda.org",
            "readthedocs.io",
            "docs.xarray.dev",  # xarray documentation
        }

    def get_staged_files(self) -> List[Path]:
        """Get list of staged files from git.

        Returns:
            List of Path objects for staged files
        """
        try:
            result = subprocess.run(
                [
                    "git",
                    "diff",
                    "--cached",
                    "--name-only",
                    "--diff-filter=ACM",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            files = [
                Path(f)
                for f in result.stdout.strip().split("\n")
                if f and self._should_check_file(Path(f))
            ]
            return files
        except subprocess.CalledProcessError as e:
            print(f"Error getting staged files: {e}", file=sys.stderr)
            return []

    def get_tracked_files(self) -> List[Path]:
        """Get list of all tracked files from git.

        Returns:
            List of Path objects for tracked files
        """
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                check=True,
            )
            files = [
                Path(f)
                for f in result.stdout.strip().split("\n")
                if f and self._should_check_file(Path(f))
            ]
            return files
        except subprocess.CalledProcessError as e:
            print(f"Error getting tracked files: {e}", file=sys.stderr)
            return []

    def _should_check_file(self, file_path: Path) -> bool:
        """Check if file should be checked based on exclusion patterns.

        Args:
            file_path: Path to check

        Returns:
            True if file should be checked, False otherwise
        """
        if not file_path.exists():
            return False

        if not file_path.is_file():
            return False

        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if file_path.match(pattern):
                return False

        # Only check text files
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(512)
                # Check if file appears to be binary
                if b"\x00" in chunk:
                    return False
        except (IOError, OSError):
            return False

        return True

    def check_absolute_paths(
        self, file_path: Path, content: str
    ) -> List[Tuple[int, str, str]]:
        """Check for absolute file system paths.

        Args:
            file_path: Path to file being checked
            content: File content

        Returns:
            List of (line_number, line_content, matched_path) tuples
        """
        issues = []
        # Patterns for absolute paths
        patterns = [
            r"/Users/[^\s\"']+",  # macOS user paths
            r"/home/[^\s\"']+",  # Linux home paths
            r"[A-Z]:\\[^\s\"']+",  # Windows paths
            r"/mnt/[^\s\"']+",  # Mounted drives
        ]

        for line_num, line in enumerate(content.split("\n"), 1):
            for pattern in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    matched_path = match.group(0)
                    # Check if path starts with allowed prefix
                    is_allowed = any(
                        matched_path.startswith(allowed)
                        for allowed in self.allowed_paths
                    )
                    if not is_allowed:
                        issues.append((line_num, line.strip(), matched_path))

        return issues

    def check_ip_addresses(
        self, file_path: Path, content: str
    ) -> List[Tuple[int, str, str]]:
        """Check for IP addresses.

        Args:
            file_path: Path to file being checked
            content: File content

        Returns:
            List of (line_number, line_content, ip_address) tuples
        """
        issues = []
        # IP address pattern (basic IPv4)
        pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"

        for line_num, line in enumerate(content.split("\n"), 1):
            matches = re.finditer(pattern, line)
            for match in matches:
                ip = match.group(0)

                # Skip version numbers (e.g., 5.2.1.1, 1.0.0.0)
                # Check for version-related context or quoted strings
                if re.search(
                    r"version|prms|v\d|refactor|changed|diff|block|restructured|['\"]"
                    + re.escape(ip)
                    + r"['\"]",
                    line.lower(),
                ):
                    continue

                # Validate it's a real IP (not version numbers, etc.)
                parts = ip.split(".")
                if all(0 <= int(p) <= 255 for p in parts):
                    # Skip obviously safe IPs
                    if not (
                        ip.startswith("127.")
                        or ip == "0.0.0.0"
                        or ip.startswith("255.")
                    ):
                        issues.append((line_num, line.strip(), ip))

        return issues

    def check_internal_hostnames(
        self, file_path: Path, content: str
    ) -> List[Tuple[int, str, str]]:
        """Check for internal server hostnames.

        Args:
            file_path: Path to file being checked
            content: File content

        Returns:
            List of (line_number, line_content, hostname) tuples
        """
        issues = []
        # Pattern for hostnames/URLs (simplified)
        pattern = r"(?:https?://)?([a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,}"

        for line_num, line in enumerate(content.split("\n"), 1):
            matches = re.finditer(pattern, line)
            for match in matches:
                hostname = match.group(0).lower()
                # Remove protocol if present
                hostname = re.sub(r"^https?://", "", hostname)

                # Skip version numbers (e.g., 1.0.dev, 0.2.0.dev0, 1.2.3.rc1)
                if re.match(r"^\d+\.\d+", hostname):
                    continue

                # Skip filenames with extensions (e.g., file.xml, test.json)
                # Real hostnames don't have common file extensions
                if re.search(
                    r"\.(xml|json|yaml|yml|txt|csv|log|md|rst|html|py|pyc|pyx|pyd|test|mock)$",
                    hostname,
                ):
                    continue

                # Skip Python code patterns (e.g., dict.items, list.keys, pytest.main)
                # Common Python attributes, methods, and testing patterns
                python_patterns = [
                    r"\.(items|keys|values|get|pop|append|extend|update|copy)\b",  # dict/list methods
                    r"\.(main|raises|mark|fixture|skip|warns|exitcode|fail|exit)\b",  # pytest
                    r"\.(assert|testing\.assert|allclose|array_equal)\b",  # numpy/xarray testing
                    r"(np|xr|pd)\.testing\.",  # numpy/xarray/pandas testing modules
                    r"\.(dev|pre|post|local)\b",  # version attributes
                    r"version\.",  # version object attributes
                    r"\.(stdev|average|repeat|mean|median|std|var)\b",  # statistics attributes
                ]
                if any(
                    re.search(pattern, hostname) for pattern in python_patterns
                ):
                    continue

                # Check if the match is part of a URL with an allowed domain
                # Look at the broader context in the line
                is_in_allowed_url = False
                for domain in self.allowed_domains:
                    # Check if this match is part of a URL containing an allowed domain
                    if domain in line.lower():
                        # If allowed domain is in the line, likely the whole URL is safe
                        is_in_allowed_url = True
                        break

                if is_in_allowed_url:
                    continue

                # Check if it's an allowed public domain
                is_allowed = any(
                    domain in hostname for domain in self.allowed_domains
                )

                # Flag if it looks like an internal server (not in allowlist)
                # and has suspicious patterns, or contains usgs/doi but isn't in allowlist
                if not is_allowed and any(
                    pattern in hostname
                    for pattern in [
                        "internal",
                        "intra",
                        "corp",
                        "dev",
                        "test",
                        "usgs",
                        "doi",
                    ]
                ):
                    issues.append((line_num, line.strip(), hostname))

        return issues

    def check_credentials(
        self, file_path: Path, content: str
    ) -> List[Tuple[int, str, str]]:
        """Check for credentials (passwords, API keys, tokens).

        Args:
            file_path: Path to file being checked
            content: File content

        Returns:
            List of (line_number, line_content, credential_type) tuples
        """
        issues = []

        # Patterns for credentials
        patterns = {
            "password": r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}",
            "api_key": r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[a-zA-Z0-9]{16,}",
            "token": r"(?i)(token|auth[_-]?token)\s*[:=]\s*['\"]?[a-zA-Z0-9]{16,}",
            "secret": r"(?i)(secret|client[_-]?secret)\s*[:=]\s*['\"]?[a-zA-Z0-9]{16,}",
            "aws_key": r"(?i)AKIA[0-9A-Z]{16}",  # AWS access key pattern
        }

        for line_num, line in enumerate(content.split("\n"), 1):
            # Skip comments that are just examples or placeholders
            if any(
                placeholder in line.lower()
                for placeholder in [
                    "example",
                    "placeholder",
                    "your_",
                    "xxx",
                    "***",
                    "<password>",
                ]
            ):
                continue

            for cred_type, pattern in patterns.items():
                if re.search(pattern, line):
                    issues.append((line_num, line.strip(), cred_type))

        return issues

    def check_file(self, file_path: Path) -> Dict[str, List[Tuple]]:
        """Check a single file for all enabled security issues.

        Args:
            file_path: Path to file to check

        Returns:
            Dictionary mapping issue type to list of issues
        """
        results = {}

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (IOError, OSError) as e:
            print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
            return results

        if self.do_check_paths:
            issues = self.check_absolute_paths(file_path, content)
            if issues:
                results["absolute_paths"] = issues

        if self.do_check_ips:
            issues = self.check_ip_addresses(file_path, content)
            if issues:
                results["ip_addresses"] = issues

        if self.do_check_hostnames:
            issues = self.check_internal_hostnames(file_path, content)
            if issues:
                results["internal_hostnames"] = issues

        if self.do_check_credentials:
            issues = self.check_credentials(file_path, content)
            if issues:
                results["credentials"] = issues

        return results

    def check_files(
        self, files: List[Path]
    ) -> Dict[Path, Dict[str, List[Tuple]]]:
        """Check multiple files.

        Args:
            files: List of file paths to check

        Returns:
            Dictionary mapping file path to results
        """
        all_results = {}

        for file_path in files:
            results = self.check_file(file_path)
            if results:
                all_results[file_path] = results

        return all_results


def format_results(
    results: Dict[Path, Dict[str, List[Tuple]]], verbose: bool = False
) -> str:
    """Format check results as a string.

    Args:
        results: Results from check_files
        verbose: Include line-by-line details

    Returns:
        Formatted string
    """
    if not results:
        return "✓ No security issues found!"

    output = []
    output.append("=" * 70)
    output.append("SECURITY CHECK RESULTS")
    output.append("=" * 70)
    output.append("")

    total_issues = sum(
        sum(len(issues) for issues in file_results.values())
        for file_results in results.values()
    )

    output.append(
        f"Found {total_issues} potential issues in {len(results)} files"
    )
    output.append("")

    for file_path, file_results in sorted(results.items()):
        output.append(f"File: {file_path}")
        output.append("-" * 70)

        for issue_type, issues in sorted(file_results.items()):
            output.append(
                f"  {issue_type.replace('_', ' ').title()}: {len(issues)} issue(s)"
            )

            if verbose:
                for line_num, line_content, detail in issues:
                    output.append(f"    Line {line_num}: {detail}")
                    output.append(f"      {line_content[:80]}")

        output.append("")

    return "\n".join(output)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for security and PII issues in repository files"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Check all tracked files (default: only staged files)",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Only check absolute paths",
    )
    parser.add_argument(
        "--check-ips",
        action="store_true",
        help="Only check IP addresses",
    )
    parser.add_argument(
        "--check-hostnames",
        action="store_true",
        help="Only check internal hostnames",
    )
    parser.add_argument(
        "--check-credentials",
        action="store_true",
        help="Only check for credentials",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed line-by-line results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write results to file instead of stdout",
    )

    args = parser.parse_args()

    # If no specific checks specified, enable all
    if not any(
        [
            args.check_paths,
            args.check_ips,
            args.check_hostnames,
            args.check_credentials,
        ]
    ):
        check_all = True
    else:
        check_all = False

    # Initialize checker
    checker = SecurityChecker(
        check_paths=check_all or args.check_paths,
        check_ips=check_all or args.check_ips,
        check_hostnames=check_all or args.check_hostnames,
        check_credentials=check_all or args.check_credentials,
    )

    # Get files to check
    if args.audit:
        print("Running full repository audit...")
        files = checker.get_tracked_files()
    else:
        print("Checking staged files...")
        files = checker.get_staged_files()

    if not files:
        print("No files to check.")
        return 0

    print(f"Checking {len(files)} files...")

    # Run checks
    results = checker.check_files(files)

    # Format and output results
    output_text = format_results(results, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Results written to {args.output}")
    else:
        print(output_text)

    # Exit with error code if issues found (for pre-commit hook)
    return 1 if results else 0


if __name__ == "__main__":
    sys.exit(main())
