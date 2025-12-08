#!/usr/bin/env node
/**
 * Parser Manager
 * Runs all available parsers in parallel and combines results
 */

const { spawn } = require('child_process');
const path = require('path');

// Read HTML from stdin
let html = '';

process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
    html += chunk;
});

process.stdin.on('end', async () => {
    const parsers = ['readability', 'defuddle', 'postlight'];
    const results = {};

    // Run all parsers in parallel
    const promises = parsers.map(parserName => {
        return new Promise((resolve) => {
            // Use .mjs for defuddle (ES module), .js for others
            const extension = parserName === 'defuddle' ? '.mjs' : '.js';
            const parserPath = path.join(__dirname, 'parsers', `${parserName}${extension}`);
            const proc = spawn('node', [parserPath]);

            let output = '';
            let errorOutput = '';

            proc.stdout.on('data', data => {
                output += data;
            });

            proc.stderr.on('data', data => {
                errorOutput += data;
                console.error(`${parserName} error:`, data.toString());
            });

            proc.on('close', (code) => {
                try {
                    if (code === 0 && output) {
                        results[parserName] = JSON.parse(output);
                    } else {
                        results[parserName] = {
                            success: false,
                            parser: parserName,
                            error: errorOutput || 'Parser failed to execute'
                        };
                    }
                } catch (e) {
                    results[parserName] = {
                        success: false,
                        parser: parserName,
                        error: 'Failed to parse output: ' + e.message
                    };
                }
                resolve();
            });

            // Send HTML to parser
            proc.stdin.write(html);
            proc.stdin.end();
        });
    });

    // Wait for all parsers to complete
    await Promise.all(promises);

    // Output combined results
    console.log(JSON.stringify({
        success: true,
        results: results,
        default: 'readability'  // Default to current parser for backward compatibility
    }, null, 2));
});
