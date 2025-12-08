#!/usr/bin/env node
/**
 * Defuddle Parser (ES Module)
 * Receives HTML via stdin, extracts article content with Defuddle, outputs JSON
 */

import { JSDOM } from 'jsdom';
import { Defuddle } from 'defuddle/node';

// Read HTML from stdin
let html = '';

process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
    html += chunk;
});

process.stdin.on('end', async () => {
    try {
        // Create JSDOM instance from HTML
        const dom = new JSDOM(html, {
            url: "https://archive.is/example"
        });

        // Extract article with Defuddle (pass JSDOM object)
        const result = await Defuddle(dom);

        if (result) {
            // Output extracted content as JSON
            const textContent = result.textContent || result.content?.replace(/<[^>]*>/g, '') || '';

            console.log(JSON.stringify({
                success: true,
                parser: 'defuddle',
                title: result.title || 'Untitled',
                byline: result.author || result.byline,
                excerpt: result.description || result.excerpt,
                textContent: textContent,
                htmlContent: result.content,
                fullTextLength: result.wordCount || textContent.length || 0,
                htmlContentLength: result.content?.length || 0,
                metadata: {
                    siteName: result.site,
                    publishedTime: result.published,
                    domain: result.domain,
                    lang: result.lang,
                    image: result.image,
                    favicon: result.favicon
                }
            }, null, 2));
        } else {
            console.log(JSON.stringify({
                success: false,
                parser: 'defuddle',
                error: "Defuddle could not extract article"
            }));
        }
    } catch (error) {
        console.log(JSON.stringify({
            success: false,
            parser: 'defuddle',
            error: error.message
        }));
    }
});
