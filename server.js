const http = require('node:http');
const https = require('node:https');
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const dns = require('node:dns').promises;

// ============================================================
// VERITY - SOURCE CREDIBILITY SERVER
// ============================================================

function loadEnvFile() {
  const envPath = path.join(__dirname, '.env');
  if (!fs.existsSync(envPath)) return;

  for (const line of fs.readFileSync(envPath, 'utf8').split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const match = trimmed.match(/^([A-Z0-9_]+)\s*=\s*(.*?)\s*$/i);
    if (!match) continue;

    const [, key, value] = match;

    if (!process.env[key]) {
      process.env[key] = value.replace(/^['"]|['"]$/g, '');
    }
  }
}

loadEnvFile();

const PORT = Number(process.env.PORT || 3000);

const API_KEY = process.env.NVIDIA_API_KEY;

const MODEL =
  process.env.NVIDIA_MODEL ||
  'nvidia/nemotron-3.5-lightning-30b-a3b';

const ROOT = __dirname;

const MAX_ARTICLE_CHARS = 24000;

const MAX_CITATION_FETCHES = 6;

const NVIDIA_HOST =
  'integrate.api.nvidia.com';

const NVIDIA_PATH =
  '/v1/chat/completions';

// ============================================================
// REPORT PROMPTS
// ============================================================

const CONTENT_PROMPT = `
You are Verity, a source credibility assessment engine.

Assess the supplied source using ONLY:
1. the supplied article text,
2. extracted source signals,
3. supplied linked-source summaries.

Do not invent facts.

Do not claim you performed a search unless a research result is explicitly supplied.

The assessment must distinguish:

1. credibility signals of the source,
2. quality of the article's evidence and claims,
3. usefulness of the source for academic research.

WRITING RULES:

- Use normal English spacing in every string.
- Never concatenate words.
- Never remove spaces between words.
- Use concise, natural English.
- Scores must be integers from 0 to 100.
- Be skeptical without being unfair.
- A small or independent source is not automatically unreliable.
- Lack of information should reduce verifiability, not automatically prove unreliability.
- Do not confuse opinion with factual evidence.
- Do not penalize a source merely because it is social media; instead classify the source type and explain its limitations.
- Primary sources should be distinguished from secondary and tertiary sources.
- A high citation count does not automatically mean high citation quality.
- A source can be useful for background even if it is not strong enough to serve as primary evidence.

Return ONLY valid JSON with this exact shape:

{
  "overall": {
    "score": 0,
    "label": "Mixed evidence"
  },

  "sourceProfile": {
    "type": "News article",
    "description": "Short description of what kind of source this is.",
    "primarySecondaryTertiary": "Secondary"
  },

  "components": [
    {
      "name": "Source Reliability",
      "score": 0,
      "reasoning": "1-2 concise sentences."
    },
    {
      "name": "Author Credibility",
      "score": 0,
      "reasoning": "1-2 concise sentences."
    },
    {
      "name": "Transparency",
      "score": 0,
      "reasoning": "1-2 concise sentences."
    },
    {
      "name": "Evidence Quality",
      "score": 0,
      "reasoning": "1-2 concise sentences."
    },
    {
      "name": "Claim Support",
      "score": 0,
      "reasoning": "1-2 concise sentences."
    },
    {
      "name": "Bias & Framing",
      "score": 0,
      "reasoning": "1-2 concise sentences."
    }
  ],

  "evidence": {
    "strength": "Moderate",
    "supportedClaims": [
      "Short supported claim or observation."
    ],
    "partiallySupportedClaims": [
      "Short partially supported claim or observation."
    ],
    "unsupportedClaims": [
      "Short unsupported or unverifiable claim or observation."
    ],
    "primarySources": 0,
    "secondarySources": 0
  },

  "framing": {
    "score": 0,
    "signals": [
      "Short framing signal or none detected."
    ],
    "summary": "1-2 concise sentences."
  },

  "recency": {
    "assessment": "Current",
    "summary": "1 concise sentence explaining whether the age of the source matters for the topic."
  },

  "researchUse": {
    "academicSuitability": 0,
    "bestUse": "Background information",
    "recommendation": "Use with caution",
    "reason": "1-2 concise sentences explaining how a student should use the source."
  },

  "facts": [
    "Short verifiable observation from the supplied material."
  ],

  "flags": [
    "Short concern or uncertainty."
  ],

  "author": {
    "name": "No author listed",
    "notes": "1-2 concise sentences."
  }
}
`;

const REPAIR_PROMPT = `
Repair the supplied JSON report.

Return ONLY valid JSON.

Do not change facts, scores, names, or meaning.

Fix malformed JSON and writing errors only.

Every word must have normal English spacing.

Never concatenate words.

Do not add new information.

Do not remove information.

Do not explain anything.
`;

// ============================================================
// BASIC HELPERS
// ============================================================

function sendEvent(response, event) {
  try {
    response.write(
      JSON.stringify(event) + '\n'
    );
  } catch {
    // Client disconnected.
  }
}

function clampScore(value, fallback = 50) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return fallback;
  }

  return Math.max(
    0,
    Math.min(
      100,
      Math.round(n)
    )
  );
}

function safeString(value, fallback = '') {
  return typeof value === 'string'
    ? value.trim()
    : fallback;
}

// ============================================================
// STATIC FILE SERVER
// ============================================================

function serveFile(response, requestPath) {
  const filename =
    requestPath === '/'
      ? 'Index.html'
      : requestPath.slice(1);

  const filePath =
    path.resolve(
      ROOT,
      filename
    );

  const relativePath =
    path.relative(
      ROOT,
      filePath
    );

  // Prevent directory traversal.
  if (
    relativePath.startsWith('..') ||
    path.isAbsolute(relativePath)
  ) {
    response.writeHead(403);
    return response.end(
      'Forbidden'
    );
  }

  if (
    !fs.existsSync(filePath) ||
    fs.statSync(filePath).isDirectory()
  ) {
    response.writeHead(404);
    return response.end(
      'Not found'
    );
  }

  const types = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css'
  };

  response.writeHead(
    200,
    {
      'Content-Type':
        `${
          types[
            path.extname(filePath)
          ] ||
          'application/octet-stream'
        }; charset=utf-8`
    }
  );

  fs.createReadStream(
    filePath
  ).pipe(response);
}

// ============================================================
// JSON REQUEST BODY
// ============================================================

function readJson(request) {
  return new Promise(
    (resolve, reject) => {
      let body = '';

      request.on(
        'data',
        chunk => {
          body += chunk.toString();

          if (
            body.length > 10000
          ) {
            reject(
              new Error(
                'Request body is too large.'
              )
            );

            request.destroy();
          }
        }
      );

      request.on(
        'end',
        () => {
          try {
            resolve(
              JSON.parse(body)
            );
          } catch {
            reject(
              new Error(
                'Invalid request JSON.'
              )
            );
          }
        }
      );

      request.on(
        'error',
        reject
      );
    }
  );
}

// ============================================================
// URL SECURITY
// ============================================================

function isPrivateAddress(
  address
) {
  if (
    net.isIP(address) === 4
  ) {
    const [
      a,
      b
    ] =
      address
        .split('.')
        .map(Number);

    return (
      a === 10 ||
      a === 127 ||
      a === 0 ||
      a >= 224 ||
      (
        a === 169 &&
        b === 254
      ) ||
      (
        a === 172 &&
        b >= 16 &&
        b <= 31
      ) ||
      (
        a === 192 &&
        b === 168
      )
    );
  }

  const lower =
    address.toLowerCase();

  return (
    lower === '::1' ||
    lower.startsWith('fc') ||
    lower.startsWith('fd') ||
    lower.startsWith('fe80:')
  );
}

async function assertPublicUrl(
  value
) {
  let parsed;

  try {
    parsed =
      new URL(value);
  } catch {
    throw new Error(
      'Please enter a valid URL.'
    );
  }

  if (
    ![
      'http:',
      'https:'
    ].includes(
      parsed.protocol
    ) ||
    parsed.username ||
    parsed.password ||
    parsed.hostname ===
      'localhost'
  ) {
    throw new Error(
      'Please enter a public http or https article URL.'
    );
  }

  let addresses;

  try {
    addresses =
      net.isIP(
        parsed.hostname
      )
        ? [
            {
              address:
                parsed.hostname
            }
          ]
        : await dns.lookup(
            parsed.hostname,
            {
              all: true
            }
          );
  } catch {
    throw new Error(
      'Could not resolve that website.'
    );
  }

  if (
    !addresses.length ||
    addresses.some(
      ({
        address
      }) =>
        isPrivateAddress(
          address
        )
    )
  ) {
    throw new Error(
      'That URL does not point to a public website.'
    );
  }

  return parsed;
}

// ============================================================
// HTML / TEXT HELPERS
// ============================================================

function decodeHtmlEntities(
  text
) {
  return text
    .replace(
      /&nbsp;/gi,
      ' '
    )
    .replace(
      /&amp;/gi,
      '&'
    )
    .replace(
      /&quot;/gi,
      '"'
    )
    .replace(
      /&#39;|&apos;/gi,
      "'"
    )
    .replace(
      /&lt;/gi,
      '<'
    )
    .replace(
      /&gt;/gi,
      '>'
    );
}

function stripTags(text) {
  return decodeHtmlEntities(
    text
      .replace(
        /<[^>]+>/g,
        ' '
      )
      .replace(
        /\s+/g,
        ' '
      )
      .trim()
  );
}

function normalizeText(
  text
) {
  return decodeHtmlEntities(
    text
      .replace(
        /\u00a0/g,
        ' '
      )
      .replace(
        /\r/g,
        ''
      )
      .replace(
        /[ \t]+/g,
        ' '
      )
      .replace(
        /\n[ \t]+/g,
        '\n'
      )
      .replace(
        /[ \t]+\n/g,
        '\n'
      )
      .replace(
        /\n{3,}/g,
        '\n\n'
      )
      .trim()
  );
}

// ============================================================
// META DATA
// ============================================================

function extractMeta(
  html,
  name
) {
  const patterns = [
    new RegExp(
      `<meta[^>]+name=["']${name}["'][^>]+content=["']([^"']+)["'][^>]*>`,
      'i'
    ),

    new RegExp(
      `<meta[^>]+property=["']${name}["'][^>]+content=["']([^"']+)["'][^>]*>`,
      'i'
    ),

    new RegExp(
      `<meta[^>]+content=["']([^"']+)["'][^>]+(?:name|property)=["']${name}["'][^>]*>`,
      'i'
    )
  ];

  for (
    const pattern of patterns
  ) {
    const match =
      html.match(
        pattern
      );

    if (
      match?.[1]
    ) {
      return stripTags(
        match[1]
      );
    }
  }

  return '';
}

function getTitle(
  html,
  fallback
) {
  const ogTitle =
    extractMeta(
      html,
      'og:title'
    );

  if (
    ogTitle
  ) {
    return ogTitle.slice(
      0,
      300
    );
  }

  const titleMatch =
    html.match(
      /<title[^>]*>([\s\S]*?)<\/title>/i
    );

  if (
    titleMatch?.[1]
  ) {
    return stripTags(
      titleMatch[1]
    ).slice(
      0,
      300
    );
  }

  return fallback;
}

function getAuthor(html) {
  const metaCandidates = [
    extractMeta(
      html,
      'author'
    ),

    extractMeta(
      html,
      'article:author'
    ),

    extractMeta(
      html,
      'parsely-author'
    )
  ];

  for (
    const candidate of metaCandidates
  ) {
    if (
      candidate &&
      candidate.length < 200
    ) {
      return candidate;
    }
  }

  const patterns = [
    /<(?:span|div|p|a)[^>]*class=["'][^"']*(?:author|byline|writer)[^"']*["'][^>]*>([\s\S]*?)<\/(?:span|div|p|a)>/i,

    /<(?:span|div|p)[^>]*id=["'][^"']*(?:author|byline|writer)[^"']*["'][^>]*>([\s\S]*?)<\/(?:span|div|p)>/i
  ];

  for (
    const pattern of patterns
  ) {
    const match =
      html.match(
        pattern
      );

    if (
      match?.[1]
    ) {
      const author =
        stripTags(
          match[1]
        )
          .replace(
            /^by\s+/i,
            ''
          )
          .trim();

      if (
        author &&
        author.length < 200
      ) {
        return author;
      }
    }
  }

  return '';
}

function getPublicationDate(
  html
) {
  const candidates = [
    extractMeta(
      html,
      'article:published_time'
    ),

    extractMeta(
      html,
      'datePublished'
    ),

    extractMeta(
      html,
      'pubdate'
    ),

    extractMeta(
      html,
      'date'
    )
  ];

  for (
    const candidate of candidates
  ) {
    if (
      candidate &&
      !Number.isNaN(
        Date.parse(candidate)
      )
    ) {
      return candidate;
    }
  }

  const timeMatch =
    html.match(
      /<time[^>]+datetime=["']([^"']+)["'][^>]*>/i
    );

  if (
    timeMatch?.[1] &&
    !Number.isNaN(
      Date.parse(
        timeMatch[1]
      )
    )
  ) {
    return timeMatch[1];
  }

  return '';
}

// ============================================================
// SOURCE TYPE DETECTION
// ============================================================

function detectSourceType(
  url,
  html,
  text
) {
  const hostname =
    url.hostname.toLowerCase();

  const pathname =
    url.pathname.toLowerCase();

  if (
    hostname.includes(
      'twitter.com'
    ) ||
    hostname.includes(
      'x.com'
    )
  ) {
    return {
      type: 'Social media post',
      family: 'Primary',
      reason:
        'The source appears to be a post hosted on X/Twitter.'
    };
  }

  if (
    hostname.includes(
      'reddit.com'
    )
  ) {
    return {
      type: 'Forum / Reddit post',
      family: 'Primary',
      reason:
        'The source appears to be a community discussion or Reddit post.'
    };
  }

  if (
    hostname.includes(
      'youtube.com'
    ) ||
    hostname.includes(
      'youtu.be'
    )
  ) {
    return {
      type: 'Video / YouTube page',
      family: 'Primary',
      reason:
        'The source appears to be a video-hosting page.'
    };
  }

  if (
    hostname.endsWith(
      '.gov'
    ) ||
    hostname.includes(
      '.gov.'
    )
  ) {
    return {
      type: 'Government source',
      family: 'Primary',
      reason:
        'The domain appears to be a government source.'
    };
  }

  if (
    hostname.endsWith(
      '.edu'
    ) ||
    hostname.includes(
      'university'
    ) ||
    hostname.includes(
      'college'
    )
  ) {
    return {
      type: 'Educational / university source',
      family: 'Primary',
      reason:
        'The domain appears to be associated with an educational institution.'
    };
  }

  if (
    pathname.endsWith(
      '.pdf'
    )
  ) {
    return {
      type: 'PDF / report',
      family: 'Secondary',
      reason:
        'The URL points to a PDF or report.'
    };
  }

  if (
    /<article\b/i.test(
      html
    )
  ) {
    return {
      type: 'Article',
      family: 'Secondary',
      reason:
        'The page contains a dedicated article element.'
    };
  }

  if (
    /\bpress release\b/i.test(
      text.slice(
        0,
        5000
      )
    )
  ) {
    return {
      type: 'Press release',
      family: 'Primary',
      reason:
        'The source appears to be a press release.'
    };
  }

  if (
    /\bblog\b/i.test(
      text.slice(
        0,
        3000
      )
    )
  ) {
    return {
      type: 'Blog / commentary',
      family: 'Secondary',
      reason:
        'The page contains signals associated with a blog or commentary format.'
    };
  }

  return {
    type: 'Webpage',
    family: 'Secondary',
    reason:
      'The page appears to be a general webpage.'
  };
}

// ============================================================
// SOURCE SIGNALS
// ============================================================

function extractSourceSignals(
  html,
  url,
  title,
  author,
  date,
  text
) {
  const lower =
    html.toLowerCase();

  const lowerText =
    text.toLowerCase();

  const externalLinks =
    [];

  const linkMatches =
    html.matchAll(
      /<a\b[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi
    );

  for (
    const match of linkMatches
  ) {
    try {
      const href =
        match[1];

      const resolved =
        new URL(
          href,
          url
        );

      if (
        ![
          'http:',
          'https:'
        ].includes(
          resolved.protocol
        )
      ) {
        continue;
      }

      if (
        resolved.hostname !==
        url.hostname
      ) {
        externalLinks.push({
          url:
            resolved.href,

          anchor:
            stripTags(
              match[2]
            ).slice(
              0,
              200
            )
        });
      }
    } catch {
      // Ignore malformed links.
    }
  }

  const citationLikeLinks =
    externalLinks.filter(
      item =>
        /doi\.org|ncbi|pubmed|gov|edu|scholar|wikipedia|sec\.gov|courtlistener/i.test(
          item.url
        ) ||
        /study|report|data|research|paper|source|evidence|citation/i.test(
          item.anchor
        )
    );

  const hasAbout =
    /\b(about us|about the author|who we are|our mission)\b/i.test(
      lowerText
    ) ||
    /\b(about)\b/i.test(
      lower
    );

  const hasContact =
    /\b(contact us|contact|email us|reach us)\b/i.test(
      lowerText
    );

  const hasCorrections =
    /\b(correction|corrections policy|errata|update policy)\b/i.test(
      lowerText
    );

  const hasDisclosure =
    /\b(disclosure|conflict of interest|funded by|sponsored by|affiliate|funding)\b/i.test(
      lowerText
    );

  const hasOpinionSignals =
    /\b(opinion|editorial|commentary|analysis|my view|i think|in my opinion)\b/i.test(
      lowerText
    );

  const hasNewsSignals =
    /\b(reported|according to|officials said|statement|investigation|reported by)\b/i.test(
      lowerText
    );

  const hasStrongLanguage =
    /\b(shocking|outrageous|insane|obviously|definitely|undeniable|everyone knows|destroyed|exposed|disaster)\b/i.test(
      lowerText
    );

  const hasFirstPerson =
    /\b(i|we|our|my)\b/i.test(
      text.slice(
        0,
        5000
      )
    );

  const wordCount =
    text
      .split(/\s+/)
      .filter(Boolean)
      .length;

  return {
    domain:
      url.hostname,

    https:
      url.protocol ===
      'https:',

    hasAuthor:
      Boolean(author),

    hasDate:
      Boolean(date),

    hasAbout,

    hasContact,

    hasCorrections,

    hasDisclosure,

    hasOpinionSignals,

    hasNewsSignals,

    hasStrongLanguage,

    hasFirstPerson,

    citationLinkCount:
      citationLikeLinks.length,

    externalLinkCount:
      externalLinks.length,

    wordCount,

    externalLinks
  };
}

// ============================================================
// HTML CLEANING
// ============================================================

function removeIrrelevantHtml(
  html
) {
  return html
    .replace(
      /<script\b[^>]*>[\s\S]*?<\/script>/gi,
      ' '
    )
    .replace(
      /<style\b[^>]*>[\s\S]*?<\/style>/gi,
      ' '
    )
    .replace(
      /<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi,
      ' '
    )
    .replace(
      /<nav\b[^>]*>[\s\S]*?<\/nav>/gi,
      ' '
    )
    .replace(
      /<footer\b[^>]*>[\s\S]*?<\/footer>/gi,
      ' '
    )
    .replace(
      /<header\b[^>]*>[\s\S]*?<\/header>/gi,
      ' '
    )
    .replace(
      /<aside\b[^>]*>[\s\S]*?<\/aside>/gi,
      ' '
    )
    .replace(
      /<form\b[^>]*>[\s\S]*?<\/form>/gi,
      ' '
    )
    .replace(
      /<svg\b[^>]*>[\s\S]*?<\/svg>/gi,
      ' '
    )
    .replace(
      /<iframe\b[^>]*>[\s\S]*?<\/iframe>/gi,
      ' '
    )
    .replace(
      /<canvas\b[^>]*>[\s\S]*?<\/canvas>/gi,
      ' '
    );
}

function extractArticleContainer(
  html
) {
  const cleaned =
    removeIrrelevantHtml(
      html
    );

  // Prefer article element.
  const articleMatch =
    cleaned.match(
      /<article\b[^>]*>([\s\S]*?)<\/article>/i
    );

  if (
    articleMatch?.[1] &&
    stripTags(
      articleMatch[1]
    ).length >= 600
  ) {
    return articleMatch[1];
  }

  // Then main.
  const mainMatch =
    cleaned.match(
      /<main\b[^>]*>([\s\S]*?)<\/main>/i
    );

  if (
    mainMatch?.[1] &&
    stripTags(
      mainMatch[1]
    ).length >= 600
  ) {
    return mainMatch[1];
  }

  // Common content containers.
  const patterns = [
    /<div\b[^>]*(?:id|class)=["'][^"']*(?:article-body|article-content|story-body|story-content|post-content|entry-content|content-body|article__body|article__content)[^"']*["'][^>]*>([\s\S]*?)<\/div>/i,

    /<section\b[^>]*(?:id|class)=["'][^"']*(?:article-body|article-content|story-body|story-content|post-content|entry-content|content-body)[^"']*["'][^>]*>([\s\S]*?)<\/section>/i
  ];

  for (
    const pattern of patterns
  ) {
    const match =
      cleaned.match(
        pattern
      );

    if (
      match?.[1] &&
      stripTags(
        match[1]
      ).length >= 600
    ) {
      return match[1];
    }
  }

  return cleaned;
}

function htmlToArticleText(
  html
) {
  let text =
    extractArticleContainer(
      html
    );

  text =
    text
      .replace(
        /<(?:br|hr)\s*\/?>/gi,
        '\n'
      )
      .replace(
        /<\/(?:p|div|section|article|h1|h2|h3|h4|h5|h6|blockquote)>/gi,
        '\n\n'
      )
      .replace(
        /<li\b[^>]*>/gi,
        '\n• '
      );

  text =
    text.replace(
      /<[^>]+>/g,
      ' '
    );

  text =
    decodeHtmlEntities(
      text
    );

  const paragraphs =
    text
      .split(
        /\n+/
      )
      .map(
        paragraph =>
          paragraph
            .replace(
              /\s+/g,
              ' '
            )
            .trim()
      )
      .filter(Boolean);

  return paragraphs.join(
    '\n\n'
  );
}

function isProbablyArticleText(
  text
) {
  if (!text) {
    return false;
  }

  const words =
    text
      .split(/\s+/)
      .filter(Boolean);

  return (
    words.length >= 250 &&
    text.length >= 1200
  );
}

// ============================================================
// FETCH ARTICLE
// ============================================================

async function fetchArticle(
  startUrl
) {
  let url =
    await assertPublicUrl(
      startUrl
    );

  for (
    let redirects = 0;
    redirects < 5;
    redirects++
  ) {
    let response;

    try {
      response =
        await fetch(
          url,
          {
            headers: {
              'User-Agent':
                'Mozilla/5.0 Verity Research Tool/1.0',

              Accept:
                'text/html,application/xhtml+xml'
            },

            redirect:
              'manual',

            signal:
              AbortSignal.timeout(
                15000
              )
          }
        );
    } catch (error) {
      throw new Error(
        `Could not download the article: ${
          error?.message ||
          'network error'
        }`
      );
    }

    // Redirect.
    if (
      response.status >= 300 &&
      response.status < 400
    ) {
      const location =
        response.headers.get(
          'location'
        );

      if (location) {
        url =
          await assertPublicUrl(
            new URL(
              location,
              url
            ).href
          );

        continue;
      }
    }

    if (!response.ok) {
      throw new Error(
        `The article site returned ${response.status}. Try a different page.`
      );
    }

    const contentType =
      response.headers.get(
        'content-type'
      ) || '';

    if (
      !contentType.includes(
        'text/html'
      )
    ) {
      throw new Error(
        'Please use an article webpage, not a PDF or file download.'
      );
    }

    const html =
      (
        await response.text()
      ).slice(
        0,
        700000
      );

    const title =
      getTitle(
        html,
        url.hostname
      );

    const author =
      getAuthor(
        html
      );

    const date =
      getPublicationDate(
        html
      );

    const text =
      normalizeText(
        htmlToArticleText(
          html
        )
      );

    if (
      !isProbablyArticleText(
        text
      )
    ) {
      throw new Error(
        'This page did not provide enough readable article text. Try opening the article directly.'
      );
    }

    const sourceType =
      detectSourceType(
        url,
        html,
        text
      );

    const signals =
      extractSourceSignals(
        html,
        url,
        title,
        author,
        date,
        text
      );

    console.log('');
    console.log(
      '=========================================='
    );
    console.log(
      'ARTICLE EXTRACTED'
    );
    console.log(
      '=========================================='
    );
    console.log(
      'Title:',
      title
    );
    console.log(
      'Author:',
      author ||
        'No author detected'
    );
    console.log(
      'Date:',
      date ||
        'No date detected'
    );
    console.log(
      'Source type:',
      sourceType.type
    );
    console.log(
      'Source family:',
      sourceType.family
    );
    console.log(
      'Characters:',
      text.length
    );
    console.log(
      'External links:',
      signals.externalLinkCount
    );
    console.log(
      'Citation-like links:',
      signals.citationLinkCount
    );
    console.log(
      'Preview:'
    );
    console.log(
      text.slice(
        0,
        900
      )
    );
    console.log(
      '=========================================='
    );
    console.log('');

    return {
      url,
      title,
      author,
      date,
      text:
        text.slice(
          0,
          MAX_ARTICLE_CHARS
        ),
      sourceType,
      signals
    };
  }

  throw new Error(
    'This URL redirected too many times.'
  );
}

// ============================================================
// LINKED SOURCE CHECKING
// ============================================================

function scoreLinkedSource(
  sourceUrl,
  title,
  text
) {
  let score = 50;

  try {
    const parsed =
      new URL(
        sourceUrl
      );

    const host =
      parsed.hostname.toLowerCase();

    if (
      host.endsWith('.gov') ||
      host.includes('.gov.')
    ) {
      score += 25;
    }

    if (
      host.endsWith('.edu') ||
      host.includes('university') ||
      host.includes('college')
    ) {
      score += 20;
    }

    if (
      host.includes('pubmed') ||
      host.includes('doi.org') ||
      host.includes('sec.gov')
    ) {
      score += 25;
    }

    if (
      /study|research|report|data|official/i.test(
        `${title} ${text.slice(0, 1000)}`
      )
    ) {
      score += 10;
    }

    if (
      /opinion|blog|commentary/i.test(
        title
      )
    ) {
      score -= 10;
    }
  } catch {
    // Keep base score.
  }

  return clampScore(
    score,
    50
  );
}

async function fetchLinkedSource(
  source
) {
  try {
    const parsed =
      await assertPublicUrl(
        source.url
      );

    const response =
      await fetch(
        parsed,
        {
          headers: {
            'User-Agent':
              'Mozilla/5.0 Verity Research Tool/1.0',

            Accept:
              'text/html,application/xhtml+xml'
          },

          redirect:
            'follow',

          signal:
            AbortSignal.timeout(
              10000
            )
        }
      );

    if (!response.ok) {
      return null;
    }

    const contentType =
      response.headers.get(
        'content-type'
      ) || '';

    if (
      !contentType.includes(
        'text/html'
      )
    ) {
      return null;
    }

    const html =
      (
        await response.text()
      ).slice(
        0,
        250000
      );

    const title =
      getTitle(
        html,
        parsed.hostname
      );

    const text =
      normalizeText(
        htmlToArticleText(
          html
        )
      ).slice(
        0,
        3000
      );

    if (
      !text ||
      text.length < 200
    ) {
      return null;
    }

    return {
      url:
        parsed.href,

      domain:
        parsed.hostname,

      title,

      anchor:
        source.anchor,

      description:
        text.slice(
          0,
          500
        ),

      score:
        scoreLinkedSource(
          parsed.href,
          title,
          text
        )
    };
  } catch {
    return null;
  }
}

async function investigateLinkedSources(
  article,
  onStatus
) {
  const links =
    article.signals.externalLinks
      .slice(
        0,
        MAX_CITATION_FETCHES
      );

  if (!links.length) {
    return [];
  }

  const results = [];

  for (
    let i = 0;
    i < links.length;
    i++
  ) {
    onStatus(
      `Checking linked evidence ${i + 1} of ${links.length}…`
    );

    const result =
      await fetchLinkedSource(
        links[i]
      );

    if (result) {
      results.push(
        result
      );
    }
  }

  return results;
}

// ============================================================
// DETERMINISTIC SIGNAL SCORES
// ============================================================

function calculateSignalScores(
  signals,
  sourceType
) {
  let sourceReliability = 50;
  let authorCredibility = 50;
  let transparency = 35;
  let evidenceQuality = 40;
  let framing = 50;

  // Source reliability.
  if (signals.https) {
    sourceReliability += 5;
  }

  if (
    sourceType.family ===
    'Primary'
  ) {
    sourceReliability += 5;
  }

  if (
    sourceType.type ===
    'Government source'
  ) {
    sourceReliability += 20;
  }

  if (
    sourceType.type ===
    'Educational / university source'
  ) {
    sourceReliability += 15;
  }

  if (
    sourceType.type ===
    'Social media post'
  ) {
    sourceReliability -= 15;
  }

  if (
    sourceType.type ===
    'Forum / Reddit post'
  ) {
    sourceReliability -= 15;
  }

  // Author.
  if (
    signals.hasAuthor
  ) {
    authorCredibility += 15;
  }

  // Transparency.
  if (
    signals.hasAuthor
  ) {
    transparency += 10;
  }

  if (
    signals.hasDate
  ) {
    transparency += 10;
  }

  if (
    signals.hasAbout
  ) {
    transparency += 10;
  }

  if (
    signals.hasContact
  ) {
    transparency += 10;
  }

  if (
    signals.hasCorrections
  ) {
    transparency += 10;
  }

  if (
    signals.hasDisclosure
  ) {
    transparency += 10;
  }

  // Evidence.
  if (
    signals.citationLinkCount > 0
  ) {
    evidenceQuality += 15;
  }

  if (
    signals.citationLinkCount >= 3
  ) {
    evidenceQuality += 10;
  }

  if (
    signals.citationLinkCount >= 6
  ) {
    evidenceQuality += 10;
  }

  // Framing.
  if (
    signals.hasOpinionSignals
  ) {
    framing -= 10;
  }

  if (
    signals.hasFirstPerson
  ) {
    framing -= 5;
  }

  if (
    signals.hasStrongLanguage
  ) {
    framing -= 15;
  }

  return {
    sourceReliability:
      clampScore(
        sourceReliability,
        50
      ),

    authorCredibility:
      clampScore(
        authorCredibility,
        50
      ),

    transparency:
      clampScore(
        transparency,
        35
      ),

    evidenceQuality:
      clampScore(
        evidenceQuality,
        40
      ),

    framing:
      clampScore(
        framing,
        50
      )
  };
}

// ============================================================
// MODEL OUTPUT CLEANING
// ============================================================

function cleanModelText(
  value
) {
  if (
    typeof value !==
    'string'
  ) {
    return value;
  }

  let text =
    value
      .replace(
        /\u00a0/g,
        ' '
      )
      .replace(
        /[\r\n\t]+/g,
        ' '
      )
      .replace(
        /\s{2,}/g,
        ' '
      )
      .trim();

  // Spaces before punctuation.
  text =
    text.replace(
      /\s+([,.;!?])/g,
      '$1'
    );

  // Missing spaces after punctuation.
  text =
    text.replace(
      /([,.;!?])([A-Za-z])/g,
      '$1 $2'
    );

  // Missing space after closing brackets.
  text =
    text.replace(
      /([)\]])([A-Za-z])/g,
      '$1 $2'
    );

  return text.trim();
}

function cleanReportText(
  value
) {
  if (
    Array.isArray(value)
  ) {
    return value.map(
      cleanReportText
    );
  }

  if (
    value !== null &&
    typeof value ===
      'object'
  ) {
    const result = {};

    for (
      const [
        key,
        item
      ] of Object.entries(
        value
      )
    ) {
      result[key] =
        cleanReportText(
          item
        );
    }

    return result;
  }

  if (
    typeof value ===
    'string'
  ) {
    return cleanModelText(
      value
    );
  }

  return value;
}

// ============================================================
// BAD SPACING DETECTION
// ============================================================

function getAllReportStrings(
  value
) {
  const strings = [];

  function walk(
    item
  ) {
    if (
      typeof item ===
      'string'
    ) {
      strings.push(
        item
      );
      return;
    }

    if (
      Array.isArray(item)
    ) {
      for (
        const child of item
      ) {
        walk(child);
      }

      return;
    }

    if (
      item !== null &&
      typeof item ===
        'object'
    ) {
      for (
        const child of Object.values(
          item
        )
      ) {
        walk(child);
      }
    }
  }

  walk(value);

  return strings;
}

function hasBadSpacing(
  report
) {
  const strings =
    getAllReportStrings(
      report
    );

  for (
    const text of strings
  ) {
    if (
      !text ||
      text.length < 25
    ) {
      continue;
    }

    const letters =
      text.match(
        /[A-Za-z]/g
      )?.length || 0;

    const spaces =
      text.match(
        / /g
      )?.length || 0;

    // Long sentence with suspiciously few spaces.
    if (
      letters > 80 &&
      spaces <
        letters / 8
    ) {
      return true;
    }

    // Common words swallowed into the next word.
    const suspicious =
      /\b(?:the|author|article|source|this|that|with|from|provides|identified|evidence|claims)[a-z]{8,}/i;

    if (
      suspicious.test(text)
    ) {
      return true;
    }
  }

  return false;
}

// ============================================================
// REPORT VALIDATION
// ============================================================

function normalizeReport(
  report,
  article,
  linkedSources,
  signalScores
) {
  report =
    cleanReportText(
      report
    );

  // Overall.
  report.overall ||= {};

  report.overall.score =
    clampScore(
      report.overall.score,
      50
    );

  report.overall.label =
    safeString(
      report.overall.label,
      'Mixed evidence'
    );

  // Source profile.
  report.sourceProfile ||= {};

  report.sourceProfile.type =
    safeString(
      report.sourceProfile.type,
      article.sourceType.type
    );

  report.sourceProfile.description =
    safeString(
      report.sourceProfile.description,
      article.sourceType.reason
    );

  report.sourceProfile.primarySecondaryTertiary =
    safeString(
      report
        .sourceProfile
        .primarySecondaryTertiary,
      article.sourceType.family
    );

  // Components.
  report.components =
    Array.isArray(
      report.components
    )
      ? report.components
      : [];

  const requiredComponents = [
    [
      'Source Reliability',
      signalScores.sourceReliability
    ],
    [
      'Author Credibility',
      signalScores.authorCredibility
    ],
    [
      'Transparency',
      signalScores.transparency
    ],
    [
      'Evidence Quality',
      signalScores.evidenceQuality
    ],
    [
      'Claim Support',
      signalScores.evidenceQuality
    ],
    [
      'Bias & Framing',
      signalScores.framing
    ]
  ];

  for (
    const [
      name,
      fallbackScore
    ] of requiredComponents
  ) {
    const existing =
      report.components.find(
        component =>
          component?.name ===
          name
      );

    if (!existing) {
      report.components.push({
        name,

        score:
          fallbackScore,

        reasoning:
          'Assessment based on the extracted source signals.'
      });
    } else {
      existing.score =
        clampScore(
          existing.score,
          fallbackScore
        );

      existing.reasoning =
        safeString(
          existing.reasoning,
          'Assessment based on the extracted source signals.'
        );
    }
  }

  // Evidence.
  report.evidence ||= {};

  report.evidence.strength =
    safeString(
      report.evidence.strength,
      'Moderate'
    );

  report.evidence.supportedClaims =
    Array.isArray(
      report
        .evidence
        .supportedClaims
    )
      ? report
          .evidence
          .supportedClaims
      : [];

  report.evidence.partiallySupportedClaims =
    Array.isArray(
      report
        .evidence
        .partiallySupportedClaims
    )
      ? report
          .evidence
          .partiallySupportedClaims
      : [];

  report.evidence.unsupportedClaims =
    Array.isArray(
      report
        .evidence
        .unsupportedClaims
    )
      ? report
          .evidence
          .unsupportedClaims
      : [];

  report.evidence.primarySources =
    Math.max(
      0,
      Number(
        report
          .evidence
          .primarySources
      ) || 0
    );

  report.evidence.secondarySources =
    Math.max(
      0,
      Number(
        report
          .evidence
          .secondarySources
      ) ||
        linkedSources.length
    );

  // Framing.
  report.framing ||= {};

  report.framing.score =
    clampScore(
      report.framing.score,
      signalScores.framing
    );

  report.framing.signals =
    Array.isArray(
      report.framing.signals
    )
      ? report.framing.signals
      : [];

  report.framing.summary =
    safeString(
      report.framing.summary,
      'No strong framing assessment was returned.'
    );

  // Recency.
  report.recency ||= {};

  report.recency.assessment =
    safeString(
      report.recency.assessment,
      article.date
        ? 'Dated source'
        : 'Date unclear'
    );

  report.recency.summary =
    safeString(
      report.recency.summary,
      article.date
        ? `Published ${new Date(
            article.date
          ).toLocaleDateString()}.`
        : 'The publication date could not be confidently determined from the page.'
    );

  // Research use.
  report.researchUse ||= {};

  report.researchUse.academicSuitability =
    clampScore(
      report
        .researchUse
        .academicSuitability,
      report.overall.score
    );

  report.researchUse.bestUse =
    safeString(
      report
        .researchUse
        .bestUse,
      'Background information'
    );

  report.researchUse.recommendation =
    safeString(
      report
        .researchUse
        .recommendation,
      'Use with caution'
    );

  report.researchUse.reason =
    safeString(
      report
        .researchUse
        .reason,
      'Use this source for context and verify important claims against stronger or primary sources when possible.'
    );

  // Facts and flags.
  report.facts =
    Array.isArray(
      report.facts
    )
      ? report.facts
      : [];

  report.flags =
    Array.isArray(
      report.flags
    )
      ? report.flags
      : [];

  // Author.
  report.author ||=
    {};

  report.author.name =
    safeString(
      report.author.name,
      article.author ||
        'No author listed'
    );

  report.author.notes =
    safeString(
      report.author.notes,
      article.author
        ? 'A byline was detected on the page, but the supplied source alone does not establish the author’s expertise.'
        : 'No author information was detected on the page.'
    );

  // Verification info.
  report.verification = {
    level:
      linkedSources.length
        ? 'Linked-source checked'
        : 'Source-page only',

    linkedSourcesChecked:
      linkedSources.length,

    note:
      linkedSources.length
        ? 'Verity checked a small sample of external sources linked from the page.'
        : 'This report is based on the source page because no linked-source check was available.'
  };

  // Basic source metadata.
  report.sourceSignals = {
    domain:
      article.signals.domain,

    https:
      article.signals.https,

    hasAuthor:
      article.signals.hasAuthor,

    hasDate:
      article.signals.hasDate,

    citationLinks:
      article.signals.citationLinkCount,

    externalLinks:
      article.signals.externalLinkCount
  };

  return cleanReportText(
    report
  );
}

// ============================================================
// NVIDIA HTTPS REQUEST
// ============================================================

function requestNvidia(body) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      ...body,
      stream: true
    });

    const options = {
      hostname: NVIDIA_HOST,
      port: 443,
      path: NVIDIA_PATH,
      method: 'POST',
      family: 4,

      headers: {
        Authorization: `Bearer ${API_KEY}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        'Content-Length': Buffer.byteLength(payload),
        'User-Agent': 'Verity/1.0'
      }
    };

    console.log('Opening streaming HTTPS connection to NVIDIA...');

    const req = https.request(options, res => {
      console.log(`NVIDIA HTTP status: ${res.statusCode}`);

      let buffer = '';
      let fullContent = '';
      let rawError = '';

      res.setEncoding('utf8');

      res.on('data', chunk => {
        buffer += chunk;

        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const rawLine of lines) {
          const line = rawLine.trim();

          if (!line) continue;

          if (!line.startsWith('data:')) {
            continue;
          }

          const data = line.slice(5).trim();

          if (data === '[DONE]') {
            continue;
          }

          let parsed;

          try {
            parsed = JSON.parse(data);
          } catch {
            continue;
          }

          if (parsed.error) {
            rawError += JSON.stringify(parsed.error);
            continue;
          }

          const choice = parsed.choices?.[0];

          if (!choice) {
            continue;
          }

          const delta = choice.delta;

          if (delta?.content) {
            fullContent += delta.content;
          }
        }
      });

      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          reject(
            new Error(
              rawError ||
              `NVIDIA API returned HTTP ${res.statusCode}.`
            )
          );

          return;
        }

        if (!fullContent.trim()) {
          reject(
            new Error(
              'NVIDIA completed the stream but returned no text.'
            )
          );

          return;
        }

        console.log(
          `NVIDIA stream complete. Received ${fullContent.length} characters.`
        );

        resolve({
          statusCode: res.statusCode,
          body: fullContent
        });
      });

      res.on('error', error => {
        reject(error);
      });
    });

    req.setTimeout(300000, () => {
      req.destroy(
        new Error(
          'NVIDIA streaming request timed out after 5 minutes.'
        )
      );
    });

    req.on('error', error => {
      console.error(
        'NVIDIA HTTPS error:',
        error
      );

      reject(error);
    });

    req.write(payload);
    req.end();
  });
}

// ============================================================
// ASK NVIDIA
// ============================================================

async function askNvidia(
  messages,
  {
    maxTokens = 3000,
    reasoningBudget = 512
  } = {}
) {
  if (!API_KEY) {
    throw new Error(
      'NVIDIA_API_KEY is not configured. Add it to .env and restart the server.'
    );
  }

  const body = {
    model: MODEL,

    messages,

    temperature: 0.1,

    max_tokens: maxTokens,

    reasoning_budget: reasoningBudget,

    chat_template_kwargs: {
      enable_thinking: true
    }
  };

  console.log(
    `Sending streaming request to NVIDIA using model: ${MODEL}`
  );

  let result;

  try {
    result = await requestNvidia(body);
  } catch (error) {
    console.error(
      'NVIDIA connection failed:',
      error
    );

    throw new Error(
      `Could not connect to NVIDIA: ${
        error?.message ||
        'unknown network error'
      }`
    );
  }

  return result.body;
}

// ============================================================
// PARSE AI REPORT
// ============================================================

function parseReport(
  content
) {
  if (
    typeof content !==
      'string' ||
    !content.trim()
  ) {
    return null;
  }

  const cleaned =
    content
      .replace(
        /```json/gi,
        ''
      )
      .replace(
        /```/g,
        ''
      )
      .trim();

  const start =
    cleaned.indexOf(
      '{'
    );

  const end =
    cleaned.lastIndexOf(
      '}'
    );

  if (
    start === -1 ||
    end === -1 ||
    end <= start
  ) {
    return null;
  }

  const candidate =
    cleaned.slice(
      start,
      end + 1
    );

  try {
    return JSON.parse(
      candidate
    );
  } catch {
    // Try common repairs.
  }

  try {
    const repaired =
      candidate
        .replace(
          /([{,]\s*)([A-Za-z_$][\w$-]*)(\s*:)/g,
          '$1"$2"$3'
        )
        .replace(
          /,\s*([}\]])/g,
          '$1'
        );

    return JSON.parse(
      repaired
    );
  } catch {
    return null;
  }
}

// ============================================================
// INVESTIGATION PIPELINE
// ============================================================

async function investigate(
  url,
  onStatus = () => {}
) {
  onStatus(
    'Downloading the source page…'
  );

  const article =
    await fetchArticle(
      url
    );

  onStatus(
    'Extracting the source structure…'
  );

  const signalScores =
    calculateSignalScores(
      article.signals,
      article.sourceType
    );

  onStatus(
    'Checking linked evidence…'
  );

  const linkedSources =
    await investigateLinkedSources(
      article,
      onStatus
    );

  const linkedSummary =
    linkedSources.length
      ? linkedSources
          .map(
            (
              source,
              index
            ) =>
              `${index + 1}. ${source.title} — ${source.domain} — ${source.description}`
          )
          .join('\n')
      : 'No external linked sources were successfully checked.';

  const sourceSignals = `
SOURCE SIGNALS:

Domain:
${article.signals.domain}

HTTPS:
${article.signals.https ? 'yes' : 'no'}

Source type:
${article.sourceType.type}

Source family:
${article.sourceType.family}

Source description:
${article.sourceType.reason}

Author detected:
${article.signals.hasAuthor ? 'yes' : 'no'}

Publication date:
${article.signals.hasDate ? article.date : 'not detected'}

About information detected:
${article.signals.hasAbout ? 'yes' : 'no'}

Contact information detected:
${article.signals.hasContact ? 'yes' : 'no'}

Corrections policy signal:
${article.signals.hasCorrections ? 'yes' : 'no'}

Funding/disclosure signal:
${article.signals.hasDisclosure ? 'yes' : 'no'}

Opinion/editorial signals:
${article.signals.hasOpinionSignals ? 'yes' : 'no'}

Strong emotional language:
${article.signals.hasStrongLanguage ? 'yes' : 'no'}

Citation-like links:
${article.signals.citationLinkCount}

External links:
${article.signals.externalLinkCount}

Word count:
${article.signals.wordCount}

LINKED SOURCES CHECKED:
${linkedSummary}
`;

  const messages = [
    {
      role:
        'system',

      content:
        CONTENT_PROMPT
    },

    {
      role:
        'user',

      content:
        `SOURCE URL:

${article.url}


TITLE:

${article.title}


AUTHOR:

${article.author || 'No author listed'}


PUBLICATION DATE:

${article.date || 'No publication date detected'}


${sourceSignals}


ARTICLE CONTENT:

${article.text}`
    }
  ];

  onStatus(
    'NVIDIA is assessing the source…'
  );

  let initial =
    await askNvidia(
      messages,
      {
        maxTokens:
          5000,

        reasoningBudget:
          700
      }
    );

  onStatus(
    'Validating the assessment…'
  );

  let report =
    parseReport(
      initial
    );

  if (
    !report ||
    hasBadSpacing(
      report
    )
  ) {
    onStatus(
      'Cleaning up the assessment…'
    );

    const repaired =
      await askNvidia(
        [
          {
            role:
              'system',

            content:
              REPAIR_PROMPT
          },

          {
            role:
              'user',

            content:
              report
                ? JSON.stringify(
                    report
                  )
                : initial
          }
        ],
        {
          maxTokens:
            5000,

          reasoningBudget:
            0
        }
      );

    const repairedReport =
      parseReport(
        repaired
      );

    if (
      repairedReport
    ) {
      report =
        repairedReport;
    }
  }

  if (!report) {
    throw new Error(
      'The model returned an invalid report. Please try the source again.'
    );
  }

  report =
    normalizeReport(
      report,
      article,
      linkedSources,
      signalScores
    );

  onStatus(
    'Preparing your results…'
  );

  return {
    content: [
      {
        type:
          'text',

        text:
          JSON.stringify(
            report
          )
      }
    ]
  };
}

// ============================================================
// HTTP SERVER
// ============================================================

const server =
  http.createServer(
    async (
      request,
      response
    ) => {
      let requestUrl;

      try {
        requestUrl =
          new URL(
            request.url,
            `http://${request.headers.host || 'localhost'}`
          );
      } catch {
        response.writeHead(
          400
        );

        return response.end(
          'Bad request'
        );
      }

      // --------------------------------------------------------
      // INVESTIGATION ENDPOINT
      // --------------------------------------------------------

      if (
        request.method ===
          'POST' &&
        requestUrl.pathname ===
          '/api/investigate'
      ) {
        response.writeHead(
          200,
          {
            'Content-Type':
              'application/x-ndjson; charset=utf-8',

            'Cache-Control':
              'no-cache, no-store, must-revalidate',

            Connection:
              'keep-alive',

            'X-Accel-Buffering':
              'no'
          }
        );

        if (!API_KEY) {
          sendEvent(
            response,
            {
              type:
                'error',

              error:
                'NVIDIA_API_KEY is not configured. Add it to .env and restart the server.'
            }
          );

          return response.end();
        }

        try {
          const body =
            await readJson(
              request
            );

          const url =
            body?.url;

          if (
            typeof url !==
              'string' ||
            !url.trim()
          ) {
            sendEvent(
              response,
              {
                type:
                  'error',

                error:
                  'A valid article URL is required.'
              }
            );

            return response.end();
          }

          const result =
            await investigate(
              url.trim(),
              message => {
                sendEvent(
                  response,
                  {
                    type:
                      'status',

                    message
                  }
                );
              }
            );

          sendEvent(
            response,
            {
              type:
                'result',

              data:
                result
            }
          );
        } catch (error) {
          console.error(
            'Investigation error:',
            error
          );

          sendEvent(
            response,
            {
              type:
                'error',

              error:
                error?.message ||
                'Could not complete the investigation.'
            }
          );
        }

        return response.end();
      }

      // --------------------------------------------------------
      // STATIC FILES
      // --------------------------------------------------------

      if (
        request.method ===
        'GET'
      ) {
        return serveFile(
          response,
          requestUrl.pathname
        );
      }

      // --------------------------------------------------------
      // Unsupported method
      // --------------------------------------------------------

      response.writeHead(
        405,
        {
          Allow:
            'GET, POST'
        }
      );

      response.end(
        'Method not allowed'
      );
    }
  );

// ============================================================
// START SERVER
// ============================================================

server.listen(
  PORT,
  () => {
    console.log('');
    console.log(
      '=============================================='
    );
    console.log(
      'VERITY SERVER STARTED'
    );
    console.log(
      '=============================================='
    );
    console.log(
      `URL: http://localhost:${PORT}`
    );
    console.log(
      `Model: ${MODEL}`
    );
    console.log(
      API_KEY
        ? 'NVIDIA API key: configured'
        : 'NVIDIA API key: MISSING'
    );
    console.log(
      'Three-layer assessment: ENABLED'
    );
    console.log(
      'Article extraction: ENABLED'
    );
    console.log(
      'Linked-source checking: ENABLED'
    );
    console.log(
      '=============================================='
    );
    console.log('');
  }
);