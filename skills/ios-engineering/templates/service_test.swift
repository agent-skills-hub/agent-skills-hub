import XCTest
@testable import <#AppModule#>

final class <#FeatureName#>ServiceTests: XCTestCase {
    func test_whenResponseIsValid_thenReturnsMappedModel() async throws {
        let client = FakeNetworkClient(result: .success(<#validResponseData#>))
        let sut = <#ServiceType#>(client: client)

        let model = try await sut.<#requestMethod#>()

        XCTAssertEqual(model, <#expectedModel#>)
    }

    func test_whenResponseIsInvalid_thenThrowsDecodingError() async {
        let client = FakeNetworkClient(result: .success(Data("invalid".utf8)))
        let sut = <#ServiceType#>(client: client)

        await XCTAssertThrowsErrorAsync(try await sut.<#requestMethod#>())
    }

    func test_whenClientFails_thenPropagatesExpectedError() async {
        let client = FakeNetworkClient(result: .failure(MockError.network))
        let sut = <#ServiceType#>(client: client)

        await XCTAssertThrowsErrorAsync(try await sut.<#requestMethod#>())
    }
}

private struct FakeNetworkClient: <#NetworkClientProtocol#> {
    let result: Result<Data, Error>

    func data(for request: URLRequest) async throws -> Data {
        _ = request
        return try result.get()
    }
}

private enum MockError: Error {
    case network
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
        XCTFail("Expected error to be thrown", file: file, line: line)
    } catch {}
}
