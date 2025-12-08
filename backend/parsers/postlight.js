#!/usr/bin/env node
/**
 * Postlight Parser (Mercury)
 * Receives HTML via stdin, extracts article content with Postlight, outputs JSON
 */

const Parser = require('@postlight/parser');

// Read HTML from stdin
let html = '';

process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
    html += chunk;
});

process.stdin.on('end', async () => {
    try {
        // Extract article with Postlight Parser
        const result = await Parser.parse('https://archive.is/example', { html });

        if (result) {
            // Output extracted content as JSON
            console.log(JSON.stringify({
                success: true,
                parser: 'postlight',
                title: result.title,
                byline: result.author,
                excerpt: result.excerpt,
                textContent: result.content?.replace(/<[^>]*>/g, ''),  // Strip HTML for text
                htmlContent: result.content,
                fullTextLength: result.word_count || 0,
                htmlContentLength: result.content?.length || 0,
                metadata: {
                    datePublished: result.date_published,
                    leadImageUrl: result.lead_image_url,
                    domain: result.domain,
                    url: result.url
                }
            }, null, 2));
        } else {
            console.log(JSON.stringify({
                success: false,
                parser: 'postlight',
                error: "Postlight could not extract article"
            }));
        }
    } catch (error) {
        console.log(JSON.stringify({
            success: false,
            parser: 'postlight',
            error: error.message
        }));
    }
});
