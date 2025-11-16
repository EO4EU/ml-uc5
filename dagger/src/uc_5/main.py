import json
import dagger
from base64 import b64encode
from typing import Annotated
from datetime import datetime
from dagger import Doc, dag, function, object_type, File, Directory

@object_type
class Uc5:
    @function
    def registry(
        self,
        bucket: Annotated[str, Doc("S3 Bucket")],
        endpoint: Annotated[str, Doc("S3 Endpoint")],
        accesskey: Annotated[dagger.Secret, Doc("S3 Access Key")],
        secretkey: Annotated[dagger.Secret, Doc("S3 Secret Key")]
    ) -> dagger.Service:
        """Start and return a registry service."""
        return (
            dag.container()
            .from_("registry:2.8.2")
            .with_env_variable("REGISTRY_HTTP_ADDR", "0.0.0.0:80")
            .with_secret_variable("REGISTRY_HTTP_SECRET", secretkey)
            .with_env_variable("REGISTRY_STORAGE", "s3")
            .with_env_variable("REGISTRY_STORAGE_S3_REGION", "default")
            .with_env_variable("REGISTRY_STORAGE_S3_BUCKET", bucket)
            .with_env_variable("REGISTRY_STORAGE_S3_REGIONENDPOINT", endpoint)
            .with_secret_variable("REGISTRY_STORAGE_S3_ACCESSKEY", accesskey)
            .with_secret_variable("REGISTRY_STORAGE_S3_SECRETKEY", secretkey)
            .with_exposed_port(80)
            .as_service()
        )

    @function
    async def build(
        self,
        bucket: Annotated[str, Doc("S3 Bucket")],
        endpoint: Annotated[str, Doc("S3 Endpoint")],
        access: Annotated[dagger.Secret, Doc("S3 Access Key")],
        secret: Annotated[dagger.Secret, Doc("S3 Secret Key")],
        repo: Annotated[str, Doc("Registry repo")],
        tag: Annotated[str, Doc("Image tag")],
        wkd: Annotated[
            dagger.Directory,
            Doc("Location of directory containing Dagger files"),
        ],
    ) -> str:
        """Build and publish image from existing Dockerfile"""
        return await (
            dag.container()
            .from_("gcr.io/kaniko-project/executor:debug")
            .with_service_binding(
                "registry.local",
                self.registry(bucket, endpoint, access, secret)
            )            
            .with_mounted_directory("/workspace", wkd)
            .with_exec(
                [
                    "/kaniko/executor",
                    "--context", 
                    "dir:///workspace/",
                    "--dockerfile",
                    "/workspace/Dockerfile",
                    "--insecure",
                    "--destination",
                    f"registry.local/{repo}:{tag}"
                ]
            )
            .stdout()
        )
    
    @function
    async def scan(
        self,
        bucket: Annotated[str, Doc("S3 Bucket")],
        endpoint: Annotated[str, Doc("S3 Endpoint")],
        access: Annotated[dagger.Secret, Doc("S3 Access Key")],
        secret: Annotated[dagger.Secret, Doc("S3 Secret Key")],
        severity: Annotated[str, Doc("Severity level")],
        exit: Annotated[str, Doc("Exit code")],
        repo: Annotated[str, Doc("Registry repo")],
        tag: Annotated[str, Doc("Image tag")],
        wkd: Annotated[
            dagger.Directory,
            Doc("Location of directory containing Dagger files"),
        ],
    ) -> dagger.File:
        """Scan image to detect vulnerabilities"""
        template = (
            dag.container()
            .from_("alpine:latest")
            .with_directory("/src", wkd)
            .with_workdir("/src")
            .with_exec(
                [
                    "wget",
                    "https://raw.githubusercontent.com/aquasecurity/trivy/refs/heads/main/contrib/html.tpl"
                ]
            )
        )
        return await (
            dag.trivy().base()
            .with_service_binding(
                "registry.local",
                self.registry(bucket, endpoint, access, secret)
            )
            .with_directory("/src", wkd)
            .with_file("/src/html.tpl", template.file("/src/html.tpl"))
            .with_exec(
                [
                    "trivy",
                    "image",
                    "--db-repository",
                    "public.ecr.aws/aquasecurity/trivy-db",
                    "--java-db-repository",
                    "public.ecr.aws/aquasecurity/trivy-java-db",
                    "--exit-code",
                    exit,
                    "--severity",
                    severity,
                    "--format",
                    "template",
                    "--template",
                    "@/src/html.tpl",
                    "--output",
                    f"/src/vulnerabilities.html",
                    "--insecure",
                    f"registry.local/{repo}:{tag}"
                ]
            )
            .file(f"/src/vulnerabilities.html")
        )

    @function
    async def sbom(
        self,
        bucket: Annotated[str, Doc("S3 Bucket")],
        endpoint: Annotated[str, Doc("S3 Endpoint")],
        access: Annotated[dagger.Secret, Doc("S3 Access Key")],
        secret: Annotated[dagger.Secret, Doc("S3 Secret Key")],
        repo: Annotated[str, Doc("Registry repo")],
        tag: Annotated[str, Doc("Image tag")],
        wkd: Annotated[
            dagger.Directory,
            Doc("Location of directory containing Dagger files"),
        ],
    ) -> dagger.File:
        """Scan image and produce SBOM report in HTML format using CycloneDX Sunshine."""
        trivy_container = (
            dag.trivy().base()
            .with_service_binding(
                "registry.local",
                self.registry(bucket, endpoint, access, secret)
            )
            .with_directory("/src", wkd)
            .with_exec(
                [
                    "trivy",
                    "image",
                    "--db-repository",
                    "public.ecr.aws/aquasecurity/trivy-db",
                    "--java-db-repository",
                    "public.ecr.aws/aquasecurity/trivy-java-db",
                    "--exit-code",
                    "0",
                    "--format",
                    "cyclonedx",
                    "--scanners",
                    "vuln",
                    "--output",
                    f"/src/sbom-report.cdx.json",
                    "--insecure",
                    f"registry.local/{repo}:{tag}"
                ]
            )
        )
        
        sunshine_container = (
            dag.container()
            .from_("python:3.11-alpine")
            .with_exec(["apk", "add", "--no-cache", "git"])
            .with_exec(["mkdir", "-p", "/reports"])
            .with_exec(["mkdir", "-p", "/app"])
            .with_exec(
                [
                    "git",
                    "clone",
                    "https://github.com/CycloneDX/Sunshine.git",
                    "/app/sunshine"
                ]
            )
            .with_workdir("/app/sunshine")
            .with_exec(
                [
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "-r",
                    "requirements.txt"
                ]
            )
            .with_file("/reports/sbom-report.cdx.json", trivy_container.file("/src/sbom-report.cdx.json"))
            .with_exec(
                [
                    "python",
                    "sunshine.py",
                    "-i",
                    "/reports/sbom-report.cdx.json",
                    "-o",
                    "/reports/sbom-report.html"
                ]
            )
        )
        
        return sunshine_container.directory("/reports")

    @function
    async def analyze_with_sonarqube(
        self,
        yaml_rules: Annotated[File, Doc("YAML file containing security rules extracted from PDF")],
        source_directory: Annotated[Directory, Doc("Source code directory to analyze")],
        sonar_host_url: Annotated[str, Doc("SonarQube server URL")],
        sonar_token: Annotated[str, Doc("SonarQube authentication token")],
        sonar_project_key: Annotated[str, Doc("SonarQube project key")],
        output_name: Annotated[str, Doc("Name for the SARIF output file")] = "sonarqube-results.sarif",
        create_quality_profile: Annotated[bool, Doc("Whether to create a custom Quality Profile based on PDF rules")] = False,
    ) -> Annotated[Directory, Doc("Directory containing SARIF JSON report and HTML report")]:
        """
        Run SonarQube compliance analysis based on PDF-extracted security rules.
        
        This function:
        1. Loads PDF security rules from YAML
        2. Optionally creates a custom SonarQube Quality Profile with only those rules
        3. Runs sonar-scanner using the custom profile (if created) or default rules
        4. Fetches analysis results from SonarQube API
        5. Generates SARIF 2.1.0 format output (JSON)
        6. Converts SARIF to HTML using sarif-tools
        
        The analysis uses rule_mapping.yaml to map PDF rules (e.g., OBJ01-J) 
        to SonarQube rules (e.g., java:S1104). This ensures compliance checking
        is based on the specific rules extracted from the PDF document.
        
        Quality Profile creation (--create-quality-profile=true):
        - Only supported for: c, cpp, java, python
        - Requires token with 'Administer Quality Profiles' permission
        - If disabled or fails, uses SonarQube default rules
        
        Requires:
        - SonarQube server to be accessible
        - Valid authentication token
        - rule_mapping.yaml with PDF->SonarQube rule mappings (if using Quality Profiles)
        
        Returns:
        - Directory with both .sarif (JSON) and .html files
        """
        
        module_source = dag.current_module().source()
        scanner_script = module_source.file("sonarqube_scanner.py")
        rule_mapping = module_source.file("rule_mapping.yaml")
        
        container = (
            dag.container()
            .from_("sonarsource/sonar-scanner-cli:latest")
            .with_user("root")
            .with_exec(["sh", "-c", "dnf install -y python3-pip"])
            .with_exec(["pip3", "install", "--no-cache-dir", "pyyaml", "requests", "sarif-tools"])
            .with_mounted_file("/workspace/rules.yaml", yaml_rules)
            .with_mounted_file("/workspace/rule_mapping.yaml", rule_mapping)
            .with_mounted_directory("/src", source_directory)
            .with_mounted_file("/workspace/sonarqube_scanner.py", scanner_script)
            .with_exec(["chown", "-R", "scanner-cli:scanner-cli", "/src"])
            .with_exec(["chmod", "-R", "u+w", "/src"])
            .with_exec(["mkdir", "-p", "/output"])
            .with_exec(["chown", "-R", "scanner-cli:scanner-cli", "/output"])
            .with_exec(["chown", "-R", "scanner-cli:scanner-cli", "/workspace"])
            .with_exec(["mkdir", "-p", "/src/target/classes"])
            .with_exec(["chown", "-R", "scanner-cli:scanner-cli", "/src/target"])
            .with_user("scanner-cli")
            .with_workdir("/src")
            .with_env_variable("SONAR_HOST_URL", sonar_host_url)
            .with_env_variable("SONAR_TOKEN", sonar_token)
            .with_env_variable("SONAR_PROJECT_KEY", sonar_project_key)
            .with_env_variable("CREATE_QUALITY_PROFILE", "true" if create_quality_profile else "false")
            .with_env_variable("PYTHONUNBUFFERED", "1")
            .with_exec([
                "python3", "/workspace/sonarqube_scanner.py",
                "/src",
                "/workspace/rules.yaml",
                f"/output/{output_name}"
            ])
            .with_exec([
                "sarif", "html",
                f"/output/{output_name}",
                "-o", f"/output/{output_name.replace('.sarif', '.html')}"
            ])
        )
        
        return container.directory("/output")

    @function
    async def synthetic_report(
        self,
        sarif_file: Annotated[File, Doc("Sonar SARIF file, e.g. sonar-report.sarif")],
        sbom_file: Annotated[File, Doc("CycloneDX SBOM JSON file, e.g. sbom-report.cdx.json")] = None,
        severity_threshold: Annotated[str, Doc("Minimum vulnerability severity to include (CRITICAL/HIGH/MEDIUM/LOW/INFO)")] = "HIGH",
    ) -> Annotated[Directory, Doc("Directory containing synthetic report HTML and JSON summary")]:
        """
        Combine SBOM CycloneDX JSON and Sonar SARIF, filter vulnerabilities above
        `severity_threshold` and error-level Sonar issues, and produce an HTML
        report plus a JSON summary. Returns a directory with
        `/output/synthetic-report.html` and `/output/synthetic-report.json`.
        """
        module_source = dag.current_module().source()
        report_script = module_source.file("generate_report.py")
        container = dag.container().from_("python:3.11-slim")
        # mount SBOM only if provided (SBOM is optional for source-only projects)
        if sbom_file is not None:
            container = container.with_mounted_file("/input/sbom.json", sbom_file)
        container = (
            container
            .with_mounted_file("/input/sonar.sarif", sarif_file)
            .with_mounted_file("/workspace/generate_report.py", report_script)
            .with_exec(["mkdir", "-p", "/output"])
            .with_env_variable("THRESHOLD", severity_threshold.upper())
        )
        # build command args dynamically so we only pass --sbom when available
        cmd = ["python3", "/workspace/generate_report.py"]
        if sbom_file is not None:
            cmd += ["--sbom", "/input/sbom.json"]
        cmd += ["--sarif", "/input/sonar.sarif", "--threshold", severity_threshold.upper(), "--outdir", "/output"]
        container = container.with_exec(cmd)
        return container.directory("/output")

    @function
    async def encode(
        self,
        registry: Annotated[str, Doc("Registry address")],
        username: Annotated[str, Doc("Registry username")],
        password: Annotated[dagger.Secret, Doc("Registry password")],
    ) -> dagger.Secret:
        """Encode username and password in base64."""
        token = await password.plaintext()
        auth_blob = b64encode(f"{username}:{token}".encode("utf-8")).decode("utf-8")

        return dagger.Client().set_secret(
            "ci_blob",
            json.dumps({"auths": {
                registry: {"auth": auth_blob},
            }})
        )

    @function
    async def push(
        self,
        bucket: Annotated[str, Doc("S3 Bucket")],
        endpoint: Annotated[str, Doc("S3 Endpoint")],
        access: Annotated[dagger.Secret, Doc("S3 Access Key")],
        secret: Annotated[dagger.Secret, Doc("S3 Secret Key")],
        registry: Annotated[str, Doc("Registry address")],
        namespace: Annotated[str, Doc("Registry namespace")],
        repo: Annotated[str, Doc("Registry repo")],
        app: Annotated[str, Doc("Application name")],
        srctag: Annotated[str, Doc("Source image tag")],
        dsttag: Annotated[str, Doc("Destination image tag")],
        username: Annotated[str, Doc("Registry username")],
        password: Annotated[dagger.Secret, Doc("Registry password")],
        wkd: Annotated[
            dagger.Directory,
            Doc("Location of directory containing Dagger files"),
        ],
    ) -> str:
        """Build and publish image from existing Dockerfile"""
        auth_blob: dagger.Secret = await self.encode(registry, username, password)
        return await (
            dag.container(platform=dagger.Platform("linux/amd64"))
            .with_service_binding(
                "registry.local",
                self.registry(bucket, endpoint, access, secret)
            )
            .from_("rapidfort/skopeo-ib:v1.16.1")
            .with_mounted_secret("/tmp/config.json", auth_blob, owner = "1000:1000")
            .with_exec(
                [
                    "skopeo",
                    "copy",
                    "--src-tls-verify=false",
                    "--src-no-creds",
                    "--dest-authfile",
                    "/tmp/config.json",
                    f"docker://registry.local/{repo}:{srctag}",
                    f"docker://{registry}/{namespace}/{repo}/{app}:{dsttag}"
                ]
            )
            .stdout()
        )
    
    @function
    async def update(
        self,
        gitlab: Annotated[str, Doc("Gitlab address")],
        repo: Annotated[str, Doc("Repository name")],
        branch: Annotated[str, Doc("Service branch")],
        username: Annotated[str, Doc("Repository username")],
        password: Annotated[dagger.Secret, Doc("Repository password")],
        wkd: Annotated[
            dagger.Directory,
            Doc("Location of directory containing Dagger files"),
        ],
    ) -> dagger.Directory:
        """Create helm chart from cookiecutter template"""
        token = await password.plaintext()
        return await (
            dag.container()
            .from_("harness/cookiecutter:latest")
            .with_directory("/src", wkd)
            .with_workdir("/src")
            .with_env_variable("GITLAB_TOKEN", token)
            .with_exec(
                [
                    "cookiecutter",
                    "--no-input",
                    "--config-file",
                    "cookiecutter-config.yaml",
                    "--checkout",
                    f"{branch}",
                    f"https://{username}:$GITLAB_TOKEN@{gitlab}/eo4eu/eo4eu-cicd/cicd-infra/cookiecutter-helm-template.git"
                ],
                expand=True
            )
            .directory(f"/src/{repo}")
        )

    @function
    async def clean(
        self,
        bucket: Annotated[str, Doc("S3 Bucket")],
        endpoint: Annotated[str, Doc("S3 Endpoint")],
        access: Annotated[dagger.Secret, Doc("S3 Access Key")],
        secret: Annotated[dagger.Secret, Doc("S3 Secret Key")],
    ) -> str:
        """Clean local registry."""
        return await (
            dag.container()
            .from_("amazon/aws-cli")
            .with_secret_variable("AWS_ACCESS_KEY_ID", access)
            .with_secret_variable("AWS_SECRET_ACCESS_KEY", secret)
            .with_exec(
                [
                    "aws",
                    "--endpoint-url",
                    f"{endpoint}",
                    "s3",
                    "rm",
                    f"s3://{bucket}/docker",
                    "--recursive"
                ]
            )
            .stdout()
        )
