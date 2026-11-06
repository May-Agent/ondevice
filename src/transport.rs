//! How the store fetches a pin's bytes.

use futures_lite::Stream;

/// An HTTP client the [`Store`](crate::Store) downloads through: a GET of a
/// URL yielding the declared length and the body as a stream of byte
/// chunks. Redirects are followed; a non-success status is an error.
///
/// The transport only moves bytes. The store owns everything else: the
/// response and idle deadlines, the size and sha256 checks, locking,
/// staging and install. The `zenwave` feature provides [`Zenwave`].
pub trait Transport: Sync {
    /// A request or body failure.
    type Error: std::error::Error + Send + Sync + 'static;
    /// One received chunk.
    type Chunk: AsRef<[u8]> + Send;
    /// The response body.
    type Body: Stream<Item = Result<Self::Chunk, Self::Error>> + Send;

    /// Sends a GET of `url` and resolves once the response head arrived.
    fn get(
        &self,
        url: &str,
    ) -> impl Future<Output = Result<Response<Self::Body>, Self::Error>> + Send;
}

/// A successful response head and its body.
#[derive(Debug)]
pub struct Response<B> {
    /// The declared `Content-Length`, when the server sent one.
    pub length: Option<u64>,
    /// The body.
    pub body: B,
}

/// The [`Transport`] over `zenwave`'s default client: the system's proxy
/// settings and trust store, redirects followed, 4xx and 5xx answered with
/// [`zenwave::Error::Http`].
#[cfg(feature = "zenwave")]
#[derive(Debug, Clone, Copy, Default)]
pub struct Zenwave;

#[cfg(feature = "zenwave")]
impl Transport for Zenwave {
    type Error = zenwave::Error;
    type Chunk = zenwave::utils::Bytes;
    type Body = futures_lite::stream::Boxed<Result<zenwave::utils::Bytes, zenwave::Error>>;

    async fn get(&self, url: &str) -> Result<Response<Self::Body>, zenwave::Error> {
        use futures_lite::StreamExt as _;
        use zenwave::Client as _;

        let mut client = zenwave::client();
        let response = client.get(url)?.await?;
        let length = response
            .headers()
            .get(zenwave::header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok()?.parse().ok());
        Ok(Response {
            length,
            body: response
                .into_body()
                .map(|chunk| chunk.map_err(zenwave::Error::from))
                .boxed(),
        })
    }
}
