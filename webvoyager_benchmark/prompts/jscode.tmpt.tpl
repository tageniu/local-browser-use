(() => {
    const elements = [];
    let index = 1;
    
    // Find all potentially interactive elements
    const selectors = 'a, button, input, select, textarea, [role="button"], [role="link"], [onclick]';
    const allElements = document.querySelectorAll(selectors);
    
    for (const el of allElements) {
        // Skip invisible elements
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
            continue;
        }
        
        // Get element info
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) continue;
        
        // Generate XPath
        function getXPath(element) {
            if (element.id) return `//*[@id="${element.id}"]`;
            if (element === document.body) return '/html/body';
            
            let ix = 0;
            const siblings = element.parentNode.childNodes;
            for (let i = 0; i < siblings.length; i++) {
                const sibling = siblings[i];
                if (sibling === element) {
                    return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                }
                if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                    ix++;
                }
            }
        }
        
        elements.push({
            index: index++,
            tag: el.tagName.toLowerCase(),
            text: el.innerText || el.value || el.placeholder || '',
            attributes: {
                id: el.id || '',
                class: el.className || '',
                href: el.href || '',
                type: el.type || '',
                placeholder: el.placeholder || '',
                value: el.value || ''
            },
            xpath: getXPath(el),
            is_visible: true,
            is_clickable: el.tagName === 'BUTTON' || el.tagName === 'A' || el.role === 'button',
            is_input: el.tagName === 'INPUT' || el.tagName === 'TEXTAREA'
        });
        
        if (elements.length >= MAX_ELEMENTS_PLACEHOLDER) break;
    }
    
    return elements;
})
