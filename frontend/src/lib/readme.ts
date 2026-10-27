import { marked } from "marked";
import DOMPurify from "dompurify";
export function renderModelReadme(
  content: string,
  model: { id: string; revision: string; readme_path: string | null },
) {
  const markdown = content.replace(
    /^\uFEFF?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/,
    "",
  );
  const fragment = DOMPurify.sanitize(
    marked.parse(markdown, { gfm: true, breaks: false, async: false }),
    {
      RETURN_DOM_FRAGMENT: true,
      ALLOWED_TAGS: [
        "p",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "a",
        "img",
        "strong",
        "b",
        "em",
        "i",
        "del",
        "s",
        "ul",
        "ol",
        "li",
        "blockquote",
        "pre",
        "code",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
        "details",
        "summary",
        "div",
        "span",
        "sup",
        "sub",
        "kbd",
        "picture",
        "source",
      ],
      ALLOWED_ATTR: [
        "href",
        "src",
        "alt",
        "title",
        "colspan",
        "rowspan",
        "start",
        "open",
      ],
    },
  );
  const repo = model.id.split("/").map(encodeURIComponent).join("/");
  const revision = encodeURIComponent(model.revision || "main");
  const directory = (model.readme_path || "README.md")
    .split("/")
    .slice(0, -1)
    .map(encodeURIComponent)
    .join("/");
  const base = `https://huggingface.co/${repo}`;
  fragment.querySelectorAll("a,img").forEach((element) => {
    const image = element.tagName === "IMG";
    const attribute = image ? "src" : "href";
    const value = element.getAttribute(attribute);
    if (!value) return;
    try {
      const urlBase =
        !image && value.startsWith("#")
          ? base
          : `${base}/${image ? "resolve" : "blob"}/${revision}/${directory ? `${directory}/` : ""}`;
      const url = new URL(value, urlBase);
      if (
        !["https:", "http:", ...(!image ? ["mailto:"] : [])].includes(
          url.protocol,
        )
      ) {
        element.removeAttribute(attribute);
        return;
      }
      element.setAttribute(attribute, url.href);
    } catch {
      element.removeAttribute(attribute);
      return;
    }
    if (element instanceof HTMLImageElement) {
      element.loading = "lazy";
      element.referrerPolicy = "no-referrer";
    } else if (element instanceof HTMLAnchorElement) {
      element.target = "_blank";
      element.rel = "noopener noreferrer";
    }
  });
  fragment.querySelectorAll("pre").forEach((e) => (e.tabIndex = 0));
  const wrapper = document.createElement("div");
  wrapper.append(fragment);
  return wrapper.innerHTML;
}
