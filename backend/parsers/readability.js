#!/usr/bin/env node
/**
 * Step 3: Test Readability.js extraction
 * Receives HTML via stdin, extracts article content, outputs JSON
 */

const { Readability } = require('@mozilla/readability');
const { JSDOM } = require('jsdom');

// Read HTML from stdin
let html = '';

process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
    html += chunk;
});

process.stdin.on('end', () => {
    try {
        // Parse HTML with JSDOM
        const dom = new JSDOM(html, {
            url: "https://archive.is/example"
        });

        // Extract article with Readability
        const reader = new Readability(dom.window.document);
        const article = reader.parse();

        if (article) {
            // Output extracted content as JSON
            const result = {
                success: true,
                parser: 'readability',
                title: article.title,
                byline: article.byline,
                excerpt: article.excerpt,
                textContent: article.textContent, // Full text (for search/metadata)
                htmlContent: article.content, // HTML with formatting preserved
                textPreview: article.textContent.substring(0, 300) + '...', // Preview for display
                fullTextLength: article.textContent.length,
                htmlContentLength: article.content.length
            };
            console.log(JSON.stringify(result, null, 2));
        } else {
            console.log(JSON.stringify({
                success: false,
                parser: 'readability',
                error: "Readability could not extract article"
            }));
        }
    } catch (error) {
        console.log(JSON.stringify({
            success: false,
            parser: 'readability',
            error: error.message
        }));
    }
});
