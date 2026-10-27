import { describe, it, expect } from "vitest";
import { renderModelReadme } from "@/lib/readme";
const model = {
  id: "owner/model",
  revision: "rev/test",
  readme_path: "docs/README.md",
};
const render = (text: string) => {
  const div = document.createElement("div");
  div.innerHTML = renderModelReadme(text, model);
  return div;
};
describe("untrusted Hub Markdown", () => {
  it("renders GFM without YAML metadata and keeps code keyboard accessible", () => {
    const doc = render(
      '---\ntags: [test]\n---\n# Model\n\n**Bold** and `inline`\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint("<safe>")\n```\n\n- One\n- Two\n\n> Note',
    );
    expect(doc.querySelector("h1")?.textContent).toBe("Model");
    expect(doc.textContent).not.toContain("tags:");
    expect(doc.querySelector("strong")?.textContent).toBe("Bold");
    expect(doc.querySelectorAll("th")).toHaveLength(2);
    expect(doc.querySelectorAll("td")).toHaveLength(2);
    expect(doc.querySelector("pre")?.tabIndex).toBe(0);
    expect(doc.querySelector("pre code")?.textContent).toContain("<safe>");
    expect(doc.querySelectorAll("li")).toHaveLength(2);
    expect(doc.querySelector("blockquote")).not.toBeNull();
  });
  it("resolves relative links, images and anchors against the exact repository revision", () => {
    const doc = render(
      "[Guide](../guide.md) ![Diagram](./images/network.png) [Section](#best-practices)",
    );
    expect(doc.querySelector("a")?.href).toBe(
      "https://huggingface.co/owner/model/blob/rev%2Ftest/guide.md",
    );
    expect(doc.querySelector("img")?.getAttribute("src")).toBe(
      "https://huggingface.co/owner/model/resolve/rev%2Ftest/docs/images/network.png",
    );
    expect(doc.querySelectorAll("a")[1].href).toBe(
      "https://huggingface.co/owner/model#best-practices",
    );
    expect(doc.querySelector("a")?.rel).toBe("noopener noreferrer");
    expect(doc.querySelector("a")?.target).toBe("_blank");
  });
  it("removes active HTML, unsafe protocols and remote styles", () => {
    const doc = render(
      '<script>window.compromised=true</script><img src="x" onerror="alert(1)"><a href="javascript:alert(1)">bad</a><iframe src="https://evil.test"></iframe><form><input></form><svg onload="alert(1)"></svg><div style="position:fixed">text</div>\n\n![bad](data:image/png;base64,abc) [local](file:///etc/passwd)',
    );
    expect(
      doc.querySelector(
        "script,iframe,form,input,svg,[onerror],[onload],[style]",
      ),
    ).toBeNull();
    expect(
      [...doc.querySelectorAll("a,img")].every(
        (e) =>
          !/^(javascript|data|file):/i.test(
            e.getAttribute("href") || e.getAttribute("src") || "",
          ),
      ),
    ).toBe(true);
  });
  it("supports empty repository documentation", () =>
    expect(render("").textContent).toBe(""));
});
