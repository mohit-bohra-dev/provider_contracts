"""Cloudflare R2 storage via S3-compatible API (boto3)."""
from __future__ import annotations

import asyncio
from urllib.parse import quote

import boto3
from botocore.client import BaseClient

from ._base import AbstractStorageProvider, StoredFile


class R2StorageProvider(AbstractStorageProvider):
    """
    S3-compatible adapter for Cloudflare R2.

    Args:
        endpoint_url: R2 endpoint (e.g. ``https://<account_id>.r2.cloudflarestorage.com``).
        access_key_id: R2 access key ID.
        secret_access_key: R2 secret access key.
        bucket_name: R2 bucket name.
        public_base_url: Optional CDN/public URL prefix for objects.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        public_base_url: str | None = None,
    ) -> None:
        self._bucket = bucket_name
        self._public_base = (public_base_url or endpoint_url).rstrip("/")
        self._client: BaseClient = boto3.client(
            "s3",
            endpoint_url=endpoint_url.rstrip("/"),
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )

    async def upload(
        self,
        data: bytes,
        key: str,
        *,
        content_type: str = "application/octet-stream",
        public: bool = False,
    ) -> StoredFile:
        _ = public

        def _put() -> None:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )

        await asyncio.to_thread(_put)
        enc = quote(key, safe="/")
        url = f"{self._public_base}/{enc}"
        return StoredFile(key=key, url=url, size_bytes=len(data), content_type=content_type)

    async def download(self, key: str) -> bytes:
        def _get() -> bytes:
            obj = self._client.get_object(Bucket=self._bucket, Key=key)
            return obj["Body"].read()  # type: ignore[no-any-return]

        return await asyncio.to_thread(_get)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(
            self._client.delete_object,
            Bucket=self._bucket,
            Key=key,
        )

    async def get_url(self, key: str, *, expires_in: int = 3600) -> str:
        _ = expires_in
        enc = quote(key, safe="/")
        return f"{self._public_base}/{enc}"
