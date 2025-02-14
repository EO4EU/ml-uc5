import dagger
from typing import Annotated
from dagger import Doc, dag, function, object_type

@object_type
class Uc5:
    @function
    def registry(
        self,
        bucket: Annotated[str, Doc("S3 Bucket")],
        endpoint: Annotated[str, Doc("S3 Endpoint")],
        accesskey: Annotated[str, Doc("S3 Access Key")],
        secretkey: Annotated[str, Doc("S3 Secret Key")]
    ) -> dagger.Service:
        """Start and return a registry service."""
        return (
            dag.container()
            .from_("registry:2")
            .with_env_variable("REGISTRY_HTTP_ADDR", "0.0.0.0:80")
            .with_env_variable("REGISTRY_HTTP_SECRET", secretkey)
            .with_env_variable("REGISTRY_STORAGE", "s3")
            .with_env_variable("REGISTRY_STORAGE_S3_REGION", "default")
            .with_env_variable("REGISTRY_STORAGE_S3_BUCKET", bucket)
            .with_env_variable("REGISTRY_STORAGE_S3_REGIONENDPOINT", endpoint)
            .with_env_variable("REGISTRY_STORAGE_S3_ACCESSKEY", accesskey)
            .with_env_variable("REGISTRY_STORAGE_S3_SECRETKEY", secretkey)
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
        accesskey = await access.plaintext()
        secretkey = await secret.plaintext()
        return await (
            dag.container()
            .from_("gcr.io/kaniko-project/executor:debug")
            .with_service_binding(
                "registry.local",
                self.registry(bucket, endpoint, accesskey, secretkey)
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
        accesskey = await access.plaintext()
        secretkey = await secret.plaintext()
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
                self.registry(bucket, endpoint, accesskey, secretkey)
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
        tag: Annotated[str, Doc("Image tag")],
        wkd: Annotated[
            dagger.Directory,
            Doc("Location of directory containing Dagger files"),
        ],
    ) -> dagger.File:
        """Scan image and produce SBOM file"""
        return await (
            dag.trivy().base()
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
                    "--output",
                    f"/src/sbom-report.cdx.json",
                    "--input",
                    f"/src/{tag}"
                ]
            )
            .file(f"/src/sbom-report.cdx.json")
        )

    @function
    async def encode(
        self,
        username: Annotated[str, Doc("Registry username")],
        password: Annotated[dagger.Secret, Doc("Registry password")],
    ) -> str:
        token = await password.plaintext()
        return await (
            dag.container()
            .from_("alpine:latest")
            .with_exec(
                [
                    "/bin/sh",
                    "-c",
                    f"printf {username}:{token} | base64"
                ]
            )
            .stdout()
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
        accesskey = await access.plaintext()
        secretkey = await secret.plaintext()
        blob = await self.encode(username, password)
        config = (
            dag.container()
            .from_("alpine:latest")
            .with_env_variable("CI_REGISTRY", registry)
            .with_env_variable("CI_BLOB", blob)
            .with_exec(
                [
                    "/bin/sh",
                    "-c",
                    "echo '{\"auths\":{\"'$CI_REGISTRY'\":{\"auth\":\"'$CI_BLOB'\"}}}' | sed 's/ //g'"
                ],
                redirect_stdout="/tmp/config.json"
            )
        )
        return await (
            dag.container(platform=dagger.Platform("linux/amd64"))
            .with_service_binding(
                "registry.local",
                self.registry(bucket, endpoint, accesskey, secretkey)
            )
            .from_("rapidfort/skopeo-ib:latest")
            .with_file("/tmp/config.json", config.file("/tmp/config.json"), owner="1000:1000")
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
            .with_exec(
                [
                    "cookiecutter",
                    "--no-input",
                    "--config-file",
                    "cookiecutter-config.yaml",
                    "--checkout",
                    f"{branch}",
                    f"https://{username}:{token}@{gitlab}/eo4eu/eo4eu-cicd/cicd-infra/cookiecutter-helm-template.git"
                ]
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
        accesskey = await access.plaintext()
        secretkey = await secret.plaintext()
        return await (
            dag.container()
            .from_("amazon/aws-cli")
            .with_env_variable("AWS_ACCESS_KEY_ID", accesskey)
            .with_env_variable("AWS_SECRET_ACCESS_KEY", secretkey)
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