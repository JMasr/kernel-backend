"""S3-compatible storage adapter. Works with MinIO (dev) and Cloudflare R2 (prod)."""
from __future__ import annotations

import asyncio
import logging
from functools import partial

import boto3
from botocore.exceptions import ClientError

from kernel_backend.core.ports.storage import StorageKeyNotFoundError, StoragePort

logger = logging.getLogger(__name__)


class S3StorageAdapter(StoragePort):
    """StoragePort adapter backed by any S3-compatible object store."""

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        region: str = "auto",
        public_endpoint_url: str = "",
    ) -> None:
        self._bucket = bucket_name
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
        )
        # Separate client for presigned URL generation so that download/upload
        # links embed the public host (e.g. https://api.kernelsecurity.tech)
        # rather than the internal Docker hostname (http://minio:9000).
        _public = public_endpoint_url or endpoint_url
        self._presign_client = (
            boto3.client(
                "s3",
                endpoint_url=_public,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region,
            )
            if _public != endpoint_url
            else self._client
        )

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    async def get(self, key: str) -> bytes:
        try:
            response = await asyncio.to_thread(
                self._client.get_object,
                Bucket=self._bucket,
                Key=key,
            )
            return await asyncio.to_thread(response["Body"].read)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise StorageKeyNotFoundError(key) from exc
            raise

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(
            self._client.delete_object,
            Bucket=self._bucket,
            Key=key,
        )

    async def presigned_upload_url(self, key: str, expires_in: int) -> str:
        return await asyncio.to_thread(
            partial(
                self._presign_client.generate_presigned_url,
                "put_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        )

    async def presigned_download_url(self, key: str, expires_in: int) -> str:
        return await asyncio.to_thread(
            partial(
                self._presign_client.generate_presigned_url,
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        )
