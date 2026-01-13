
const BACKEND_URL = 'http://localhost:8000'; // Default to local backend
const API_BASE = `${BACKEND_URL}/api`;

document.addEventListener('DOMContentLoaded', () => {
    // Determine which page we are on
    const path = window.location.pathname;
    const isAnalyzePage = path.includes('analyze.html');
    const isDownloadsPage = path.includes('downloads.html');
    const isGalleryPage = path.includes('gallery.html');

    // Navigation Active State
    const navButtons = document.querySelectorAll('.nav-tab');
    navButtons.forEach(btn => {
        const href = btn.getAttribute('href');
        if (path === '/' && href === '/') btn.classList.add('active');
        else if (href !== '/' && path.includes(href)) btn.classList.add('active');
        else btn.classList.remove('active');
    });

    // Page Specific Initialization
    if (isAnalyzePage) {
        const urlParams = new URLSearchParams(window.location.search);
        const taskId = urlParams.get('task');
        if (taskId) pollStatus(taskId);
        else window.location.href = 'index.html';
    } else if (isDownloadsPage) {
        loadDownloads();
    } else if (isGalleryPage) {
        loadGallery();
    }

    // Analyze Button (Home Page)
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeVideo);
    }

    // Clear Button (Home Page)
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            const urlInput = document.getElementById('video-url');
            if (urlInput) {
                urlInput.value = '';
                urlInput.focus();
            }
        });
    }
});

async function analyzeVideo() {
    const urlInput = document.getElementById('video-url');
    const durationSelect = document.getElementById('duration-select');

    if (!urlInput.value) {
        alert('Please enter a YouTube URL');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                urls: [urlInput.value],
                duration_mode: durationSelect ? durationSelect.value : '30'
            })
        });

        if (!response.ok) throw new Error('Analysis failed to start');

        const data = await response.json();
        const taskId = data.task_ids[0];

        // Redirect to analysis page with task ID
        window.location.href = `analyze.html?task=${taskId}`;

    } catch (error) {
        console.error(error);
        alert('Error: ' + error.message);
    }
}

function pollStatus(taskId) {
    const progressBar = document.getElementById('progress-bar');
    const statusText = document.getElementById('status-text');

    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/status/${taskId}`);
            const task = await response.json();

            if (progressBar) progressBar.style.width = `${task.progress}%`;
            if (statusText) statusText.innerText = `Status: ${task.status} (${task.progress}%)`;

            if (task.status === 'completed') {
                clearInterval(interval);
                displayResults(task);
            } else if (task.status === 'failed') {
                clearInterval(interval);
                if (statusText) statusText.innerText = `Failed: ${task.error}`;
            }

        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 1000);
}

function displayResults(task) {
    const resultsContainer = document.getElementById('results-container');
    const clipsList = document.getElementById('clips-list');

    if (resultsContainer) resultsContainer.classList.remove('hidden');
    if (clipsList) clipsList.innerHTML = '';

    if (task.segments && task.segments.length > 0) {
        task.segments.forEach((seg, index) => {
            // Using logic to match style.css .timeline-row structure
            const row = document.createElement('div');
            row.className = 'timeline-row';
            row.innerHTML = `
                <div class="timeline-info">
                    <span class="timeline-title">${seg.title}</span>
                    <span class="timeline-meta">Start: ${formatTime(seg.start)} | End: ${formatTime(seg.end)} | Score: ${seg.viral_score}</span>
                </div>
                <div class="timeline-actions">
                    <select id="quality-${index}" class="quality-select" style="margin-right: 0.5rem;">
                        <option value="360">360p</option>
                        <option value="480">480p</option>
                        <option value="720">720p</option>
                        <option value="1080" selected>1080p</option>
                    </select>
                    <button class="action-btn" onclick="previewClip('${task.video_id}', ${seg.start}, ${seg.end})" style="margin-right: 0.5rem;">
                        Preview
                    </button>
                    <button class="action-btn" onclick="downloadClip('${task.video_id}', ${seg.start}, ${seg.end}, '${seg.title}', ${index})">
                        Download
                    </button>
                </div>
            `;
            if (clipsList) clipsList.appendChild(row);
        });
    } else {
        if (clipsList) clipsList.innerHTML = '<p style="color:var(--text-muted)">No viral segments found.</p>';
    }
}

async function generateClip(videoId, start, end, title, quality = '1080') {
    try {
        const response = await fetch(`${API_BASE}/generate-clip`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: videoId,
                start: start,
                end: end,
                title: title,
                quality: quality
            })
        });

        if (!response.ok) throw new Error('Generation failed');

        const data = await response.json();

        // Create a temporary link and trigger download
        const link = document.createElement('a');
        const downloadUrl = data.url.startsWith('http') ? data.url : `${BACKEND_URL}${data.url}`;
        link.href = downloadUrl;
        link.download = `${title.replace(/[^a-z0-9]/gi, '_')}_${start}_${end}_${quality}p.mp4`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

    } catch (error) {
        alert('Failed to generate clip: ' + error.message);
    }
}

async function loadDownloads() {
    const list = document.getElementById('downloads-list');
    if (!list) return;
    list.innerHTML = '<p style="color:var(--text-muted)">Loading...</p>';

    try {
        const response = await fetch(`${API_BASE}/processed-videos`);
        const data = await response.json();

        list.innerHTML = '';
        data.videos.forEach(video => {
            const item = document.createElement('div');
            // Matching style.css .download-item
            item.className = 'download-item';
            item.innerHTML = `
                <div class="download-info">
                    <span class="download-title">${video.filename}</span>
                    <span class="download-meta">Size: ${(video.size / 1024 / 1024).toFixed(1)} MB</span>
                </div>
                <div class="download-actions">
                    <a href="${video.url.startsWith('http') ? video.url : BACKEND_URL + video.url}" download class="action-btn" style="text-decoration:none">Download</a>
                </div>
            `;
            list.appendChild(item);
        });

        if (data.videos.length === 0) list.innerHTML = '<p style="color:var(--text-muted)">No downloads available.</p>';

    } catch (error) {
        list.innerHTML = '<p style="color:red">Error loading downloads.</p>';
    }
}

async function loadGallery() {
    const grid = document.getElementById('gallery-grid');
    if (!grid) return;
    grid.innerHTML = '<p style="color:var(--text-muted)">Loading...</p>';

    try {
        const response = await fetch(`${API_BASE}/processed-videos`);
        const data = await response.json();

        grid.innerHTML = '';
        data.videos.forEach(video => {
            const card = document.createElement('div');
            // Matching style.css .gallery-item (not .gallery-card)
            card.className = 'gallery-item';
            card.innerHTML = `
                <div class="gallery-icon">🎬</div>
                <span class="gallery-title">${video.filename}</span>
                <a href="${video.url.startsWith('http') ? video.url : BACKEND_URL + video.url}" target="_blank" class="action-btn" style="text-decoration:none; display:inline-block; margin-top:0.5rem">Watch</a>
            `;
            grid.appendChild(card);
        });

        if (data.videos.length === 0) grid.innerHTML = '<p style="color:var(--text-muted)">Gallery is empty.</p>';

    } catch (error) {
        grid.innerHTML = '<p style="color:red">Error loading gallery.</p>';
        console.error(error);
    }
}

function downloadClip(videoId, start, end, title, index) {
    const qualitySelect = document.getElementById(`quality-${index}`);
    const quality = qualitySelect ? qualitySelect.value : '1080';
    generateClip(videoId, start, end, title, quality);
}

async function previewClip(videoId, start, end) {
    try {
        // Create preview modal
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;

        const content = document.createElement('div');
        content.style.cssText = `
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 20px;
            max-width: 800px;
            width: 90%;
            max-height: 80vh;
            overflow: hidden;
        `;

        content.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: var(--text-main);">Preview: ${formatTime(start)} - ${formatTime(end)}</h3>
                <button onclick="closePreviewModal()" style="
                    background: var(--primary);
                    border: none;
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 600;
                ">Close</button>
            </div>
            <video controls style="width: 100%; max-height: 60vh; border-radius: 10px;" autoplay>
                <source src="${API_BASE}/preview/${videoId}?start=${start}&duration=${end - start}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        `;

        modal.appendChild(content);
        document.body.appendChild(modal);

        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });

    } catch (error) {
        alert('Failed to load preview: ' + error.message);
    }
}

function closePreviewModal() {
    const modal = document.querySelector('div[style*="position: fixed"]');
    if (modal) {
        modal.remove();
    }
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}
